"""
Video streaming module using local OpenCV + YOLOv8 for real-time PPE detection.
Optimized for low latency and high performance.
"""
import base64
import cv2
import numpy as np
import time
from typing import Optional
from datetime import datetime
from threading import Thread, Event, Lock

# Enable OpenCV optimizations globally
cv2.setUseOptimized(True)

class VideoStreamProcessor:
    """Handles real-time video streaming with optimized PPE and Face detection."""

    def __init__(self, detector=None, max_fps: int = 25):
        self.detector = detector
        self.max_fps = max_fps
        
        self.is_running = False
        self.stop_event = Event()

        # Thread-safe buffers for latest frame and results
        self.latest_frame_b64 = None
        self.latest_result = None
        self.frame_id = 0
        self.last_served_frame_id = -1
        self._buffer_lock = Lock()

        # Thread-safe variables for non-blocking ML inference
        self.frame_for_inference = None
        self.latest_detections = {"ppe": [], "faces": [], "violations": [], "summary": {}}
        self._inference_lock = Lock()

        # Colors for bounding boxes (BGR)
        self.colors = {
            "Helmet": (0, 200, 0), "Vest": (0, 255, 0), "Goggles": (0, 180, 80),
            "Mask": (0, 140, 100), "Safety Shoes": (0, 120, 80), "Gloves": (60, 160, 0),
            "NO Helmet": (0, 0, 255), "NO Vest": (0, 60, 255), "NO Goggles": (80, 50, 255),
            "NO Mask": (120, 100, 255), "NO Safety Shoes": (140, 120, 255), "NO Gloves": (80, 80, 255),
            "face": (0, 255, 255), "default": (128, 128, 128),
        }

    def _add_overlays(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Add timestamp and compliance status overlays."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        violations = detections.get("violations", [])
        status_text = "COMPLIANT" if not violations else f"{len(violations)} VIOLATION(S)"
        status_color = (0, 255, 0) if not violations else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        return frame

    def _capture_loop(self, video_source):
        """
        Capture loop: reads frames at full speed and annotates with LATEST detections.
        ML inference runs in a SEPARATE background thread (_inference_loop).
        This ensures smooth video at full FPS regardless of ML speed.
        """
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW) if isinstance(video_source, int) else cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera source {video_source}")
            self.is_running = False
            return

        # Latency optimization: minimal buffer, MJPEG codec
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        frame_interval = 1.0 / self.max_fps
        frame_count = 0
        
        print(f"Stream started at {self.max_fps} FPS. Source: {video_source}")

        try:
            while not self.stop_event.is_set():
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if frame.shape[1] != 640 or frame.shape[0] != 360:
                    frame = cv2.resize(frame, (640, 360))

                frame_count += 1

                if self.detector:
                    # Every 4th frame: hand off to background inference thread (non-blocking)
                    if frame_count % 4 == 0:
                        with self._inference_lock:
                            if self.frame_for_inference is None:
                                self.frame_for_inference = frame.copy()
                    
                    # Always use the LATEST detections from the background thread
                    with self._inference_lock:
                        cached_detections = self.latest_detections

                    # Fast OpenCV annotation on every frame
                    annotated = self._annotate_cv2(frame, cached_detections)
                else:
                    cached_detections = {"ppe": [], "faces": [], "violations": [], "summary": {}}
                    annotated = frame.copy()

                annotated = self._add_overlays(annotated, cached_detections)

                # JPEG encode
                _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                # Full result payload for WebSocket clients
                result = {
                    "success": True,
                    "detections": cached_detections,
                    "violations": cached_detections.get("violations", []),
                    "faces": cached_detections.get("faces", []),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "compliant": len(cached_detections.get("violations", [])) == 0,
                }

                # Update thread-safe buffer
                with self._buffer_lock:
                    self.latest_frame_b64 = frame_b64
                    self.latest_result = result
                    self.frame_id += 1

                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        finally:
            cap.release()
            self.is_running = False

    def _annotate_cv2(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Fast OpenCV annotation - draws bounding boxes for PPE and faces."""
        img = frame.copy()
        try:
            for det in detections.get("ppe", []):
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = det["label"]
                color = self.colors.get(label, self.colors["default"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for face in detections.get("faces", []):
                x1, y1, x2, y2 = map(int, face["bbox"])
                name = face.get("name") or "Unknown"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        except Exception as e:
            print(f"Error drawing boxes: {e}")
        return img

    def _inference_loop(self):
        """
        Background ML inference thread.
        Runs independently from the capture loop so video is always smooth.
        Picks up a frame from the queue, runs the detector, stores the result.
        """
        while not self.stop_event.is_set():
            frame_to_process = None
            with self._inference_lock:
                if self.frame_for_inference is not None:
                    frame_to_process = self.frame_for_inference
                    self.frame_for_inference = None

            if frame_to_process is not None and self.detector:
                try:
                    _, new_detections = self.detector.process_image(frame_to_process)
                    with self._inference_lock:
                        self.latest_detections = new_detections
                except Exception as e:
                    print(f"[Inference ERROR]: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                time.sleep(0.01)

    def start(self, video_source: int = 0) -> bool:
        if self.is_running:
            return True
        self.stop_event.clear()
        self.is_running = True
        Thread(target=self._capture_loop, args=(video_source,), daemon=True).start()
        if self.detector:
            Thread(target=self._inference_loop, daemon=True).start()
            print("Background ML inference thread started.")
        return True

    def stop(self):
        self.stop_event.set()
        self.is_running = False

    def get_frame_nowait(self) -> Optional[dict]:
        """Get the latest frame if it's new (non-blocking)."""
        with self._buffer_lock:
            if self.frame_id > self.last_served_frame_id and self.latest_frame_b64:
                self.last_served_frame_id = self.frame_id
                return {"frame": self.latest_frame_b64, "result": self.latest_result}
        return None

    def get_latest_frame(self) -> Optional[dict]:
        """Get the latest frame regardless of whether it was already served."""
        with self._buffer_lock:
            if self.latest_frame_b64:
                return {"frame": self.latest_frame_b64, "result": self.latest_result}
        return None


# Global Singleton
_processor: Optional[VideoStreamProcessor] = None

def get_video_processor(detector=None) -> VideoStreamProcessor:
    global _processor
    if _processor is None:
        _processor = VideoStreamProcessor(detector=detector)
    return _processor
