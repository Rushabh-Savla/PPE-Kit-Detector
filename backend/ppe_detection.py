"""
PPE Safety Detection Service
Based on: https://github.com/prodbykosta/ppe-safety-detection-ai
Detects: Helmets and Safety Vests using YOLO-based model via Roboflow
"""
import asyncio
import base64
import json
import threading
import time
from io import BytesIO
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Try to import inference SDK
try:
    from inference_sdk import InferenceHTTPClient
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: inference_sdk not available. Install with: pip install inference-sdk")

# ================= CONFIGURATION =================
# PPE Detection Model from prodbykosta/ppe-safety-detection-ai
# Uses Roboflow's PPE detection model
ROBOFLOW_API_KEY = "e96XmSjaw1rrek7TBBxu"
ROBOFLOW_MODEL_ID = "ppe-object-detection-yolov8/3"  # PPE detection model

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# ================= GLOBAL STATE =================
current_detections = {
    "helmet": {"detected": False, "confidence": 0.0, "bbox": None},
    "vest": {"detected": False, "confidence": 0.0, "bbox": None},
    "persons": 0,
    "compliant": False,
    "timestamp": None,
    "frame_base64": None
}

# Temporal smoothing - track detections over time
detection_history = {
    "helmet": [],
    "vest": []
}
SMOOTHING_WINDOW = 3  # Number of frames for temporal smoothing

# Global event loop reference
main_loop = None


# ================= PPE DETECTOR CLASS =================
class PPEDetector:
    """PPE Detection using Roboflow inference."""

    def __init__(self):
        self.client = None
        self.model_id = ROBOFLOW_MODEL_ID

        if INFERENCE_AVAILABLE:
            print("Initializing Roboflow PPE Detection client...")
            self.client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=ROBOFLOW_API_KEY
            )
            print(f"PPE Detection model: {self.model_id}")
        else:
            print("ERROR: inference_sdk not available!")

    def detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Run PPE detection on image.
        Returns detection results for helmet and vest.
        """
        global detection_history

        result = {
            "helmet": {"detected": False, "confidence": 0.0, "bbox": None},
            "vest": {"detected": False, "confidence": 0.0, "bbox": None},
            "persons": 0,
            "compliant": False,
            "raw_detections": [],
            "timestamp": time.time()
        }

        if not self.client:
            return result

        try:
            # Convert bytes to base64 for Roboflow
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Run inference
            response = self.client.infer(
                image_base64,
                model_id=self.model_id
            )

            predictions = response.get('predictions', [])

            # Process each detection
            for pred in predictions:
                label = pred.get('class', '').lower()
                confidence = pred.get('confidence', 0.0)

                # Get bounding box
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)
                bbox = [
                    int(x - width/2),
                    int(y - height/2),
                    int(x + width/2),
                    int(y + height/2)
                ]

                raw_detection = {
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox
                }
                result["raw_detections"].append(raw_detection)

                # Map detections to our categories
                if confidence >= CONFIDENCE_THRESHOLD:
                    if 'helmet' in label or 'hardhat' in label or 'hard-hat' in label:
                        if confidence > result["helmet"]["confidence"]:
                            result["helmet"] = {
                                "detected": True,
                                "confidence": confidence,
                                "bbox": bbox
                            }

                    elif 'vest' in label or 'jacket' in label or 'safety-vest' in label:
                        if confidence > result["vest"]["confidence"]:
                            result["vest"] = {
                                "detected": True,
                                "confidence": confidence,
                                "bbox": bbox
                            }

                    elif 'person' in label or 'worker' in label:
                        result["persons"] += 1

            # Apply temporal smoothing
            detection_history["helmet"].append(result["helmet"]["detected"])
            detection_history["vest"].append(result["vest"]["detected"])

            # Keep only last N frames
            detection_history["helmet"] = detection_history["helmet"][-SMOOTHING_WINDOW:]
            detection_history["vest"] = detection_history["vest"][-SMOOTHING_WINDOW:]

            # Smoothed detection - require detection in majority of recent frames
            helmet_smoothed = sum(detection_history["helmet"]) > SMOOTHING_WINDOW / 2
            vest_smoothed = sum(detection_history["vest"]) > SMOOTHING_WINDOW / 2

            # Override with smoothed values
            if helmet_smoothed and not result["helmet"]["detected"]:
                result["helmet"]["detected"] = True
                result["helmet"]["confidence"] = 0.5  # Lower confidence for smoothed

            if vest_smoothed and not result["vest"]["detected"]:
                result["vest"]["detected"] = True
                result["vest"]["confidence"] = 0.5

            # Determine compliance (both helmet and vest detected)
            result["compliant"] = result["helmet"]["detected"] and result["vest"]["detected"]

            print(f"PPE Detection: Helmet={result['helmet']['detected']} ({result['helmet']['confidence']:.2f}), "
                  f"Vest={result['vest']['detected']} ({result['vest']['confidence']:.2f}), "
                  f"Compliant={result['compliant']}")

        except Exception as e:
            print(f"PPE Detection error: {e}")
            import traceback
            traceback.print_exc()

        return result


# Initialize detector
detector = PPEDetector()


# ================= WEBSOCKET MANAGER =================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                self.disconnect(connection)


manager = ConnectionManager()
app = FastAPI(title="PPE Safety Detection Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= API ENDPOINTS =================

@app.get("/")
async def root():
    return {
        "service": "PPE Safety Detection",
        "model": ROBOFLOW_MODEL_ID,
        "detects": ["helmet", "vest"],
        "status": "running"
    }


@app.get("/status")
async def get_status():
    """Get current detection status."""
    return current_detections


@app.post("/detect")
async def detect_ppe(file: UploadFile = File(...)):
    """
    Run PPE detection on uploaded image.
    Returns helmet and vest detection results.
    """
    global current_detections

    try:
        # Read image
        image_bytes = await file.read()

        # Run detection
        result = detector.detect(image_bytes)

        # Update global state
        current_detections.update({
            "helmet": result["helmet"],
            "vest": result["vest"],
            "persons": result["persons"],
            "compliant": result["compliant"],
            "timestamp": result["timestamp"]
        })

        # Broadcast to WebSocket clients
        if main_loop and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(current_detections),
                main_loop
            )

        return {
            "success": True,
            "detections": {
                "helmet": result["helmet"],
                "vest": result["vest"],
                "persons": result["persons"],
                "compliant": result["compliant"],
                "raw": result["raw_detections"]
            }
        }

    except Exception as e:
        print(f"Detection error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/detect-frame")
async def detect_frame_base64(data: dict):
    """
    Run PPE detection on base64-encoded frame.
    Expected: { "frame": "base64_encoded_image_data" }
    """
    global current_detections

    try:
        frame_base64 = data.get("frame", "")

        # Remove data URL prefix if present
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(frame_base64)

        # Run detection
        result = detector.detect(image_bytes)

        # Update global state
        current_detections.update({
            "helmet": result["helmet"],
            "vest": result["vest"],
            "persons": result["persons"],
            "compliant": result["compliant"],
            "timestamp": result["timestamp"],
            "frame_base64": frame_base64[:100] + "..."  # Truncated for logging
        })

        return {
            "success": True,
            "detections": {
                "helmet": result["helmet"],
                "vest": result["vest"],
                "persons": result["persons"],
                "compliant": result["compliant"]
            }
        }

    except Exception as e:
        print(f"Frame detection error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time detection updates."""
    await manager.connect(websocket)

    # Send current state on connect
    await websocket.send_json(current_detections)

    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "frame":
                    frame_base64 = message.get("frame", "")

                    # Remove data URL prefix if present
                    if "," in frame_base64:
                        frame_base64 = frame_base64.split(",")[1]

                    # Decode and detect
                    image_bytes = base64.b64decode(frame_base64)
                    result = detector.detect(image_bytes)

                    # Update state and broadcast
                    current_detections.update({
                        "helmet": result["helmet"],
                        "vest": result["vest"],
                        "persons": result["persons"],
                        "compliant": result["compliant"],
                        "timestamp": result["timestamp"]
                    })

                    # Send result back to this client
                    await websocket.send_json({
                        "type": "detection",
                        "data": current_detections
                    })

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/reset")
async def reset_detections():
    """Reset detection state."""
    global current_detections, detection_history

    current_detections = {
        "helmet": {"detected": False, "confidence": 0.0, "bbox": None},
        "vest": {"detected": False, "confidence": 0.0, "bbox": None},
        "persons": 0,
        "compliant": False,
        "timestamp": None,
        "frame_base64": None
    }

    detection_history = {
        "helmet": [],
        "vest": []
    }

    return {"status": "reset", "detections": current_detections}


# ================= STARTUP EVENT =================

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    print("PPE Detection Service started!")
    print(f"Model: {ROBOFLOW_MODEL_ID}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")


# ================= MAIN =================

if __name__ == "__main__":
    print("Starting PPE Safety Detection Service...")
    print("Endpoints:")
    print("  - POST /detect - Upload image for detection")
    print("  - POST /detect-frame - Base64 frame detection")
    print("  - GET /status - Current detection status")
    print("  - WS /ws - WebSocket for real-time detection")

    uvicorn.run(app, host="0.0.0.0", port=8002)
