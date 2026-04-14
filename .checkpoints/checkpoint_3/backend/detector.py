"""
Detection module combining PPE detection, face detection, and face recognition.
Uses Roboflow workflow for PPE detection with SAHI (Slicing Aided Hyper Inference).
"""
import pickle
import base64
import tempfile
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Try to import face recognition components
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


class PersonDetector:
    """Combined PPE and face detection/recognition using Roboflow."""

    def __init__(self, known_faces_dir: str = "../known_faces"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # PPE Detection Model 1: keremberke/yolov8m-protective-equipment-detection
        print("Loading PPE detection model 1...")
        try:
            ppe_model1_path = hf_hub_download(
                repo_id="keremberke/yolov8m-protective-equipment-detection",
                filename="best.pt"
            )
            self.ppe_model1 = YOLO(ppe_model1_path).to(self.device)
            print("PPE detection model 1 loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load PPE model 1: {e}")
            self.ppe_model1 = None

        # PPE Detection Model 2: Tanishjain9/yolov8n-ppe-detection-6classes
        print("Loading PPE detection model 2...")
        try:
            ppe_model2_path = hf_hub_download(
                repo_id="Tanishjain9/yolov8n-ppe-detection-6classes",
                filename="best.pt"
            )
            self.ppe_model2 = YOLO(ppe_model2_path).to(self.device)
            print("PPE detection model 2 loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load PPE model 2: {e}")
            self.ppe_model2 = None

        # Face Detection Model (arnabdhar/YOLOv8-Face-Detection)
        print("Loading face detection model...")
        face_model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt"
        )
        self.face_model = YOLO(face_model_path).to(self.device)

        # Face Recognition (using FaceNet)
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(exist_ok=True)
        self.known_faces_file = self.known_faces_dir / "embeddings.pkl"

        if FACE_RECOGNITION_AVAILABLE:
            print("Loading face recognition model...")
            # MTCNN with lower thresholds for better detection
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                thresholds=[0.5, 0.6, 0.6],
                min_face_size=20,
                post_process=True
            )
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.known_faces = self._load_known_faces()
        else:
            print("Face recognition not available (facenet-pytorch not installed)")
            self.mtcnn = None
            self.facenet = None
            self.known_faces = {}

        # Colors for different PPE classes (green = good, red = missing/violation)
        self.colors = {
            # Positive detections (wearing PPE) - Green shades
            "helmet": (0, 200, 0),
            "Helmet": (0, 200, 0),
            "goggles": (0, 180, 80),
            "Goggles": (0, 180, 80),
            "mask": (0, 140, 100),
            "Mask": (0, 140, 100),
            "shoes": (0, 120, 80),
            "Safety Shoe": (0, 120, 80),
            "Safety Shoes": (0, 120, 80),
            "Vest": (0, 255, 0),
            "vest": (0, 255, 0),
            # Missing PPE - Red shades
            "Without Helmet": (255, 0, 0),
            "Without Goggles": (255, 50, 50),
            "Without Mask": (255, 100, 100),
            "Without Safety Shoes": (255, 120, 120),
            "Without Vest": (255, 60, 60),
            # Face
            "face": (255, 255, 0),
            "default": (128, 128, 128)
        }

        print("All models loaded successfully!")

    def _load_known_faces(self) -> dict:
        """Load known face embeddings from disk."""
        if self.known_faces_file.exists():
            with open(self.known_faces_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def reload_faces(self):
        """Reload known faces from disk. Call this after new face registration."""
        if FACE_RECOGNITION_AVAILABLE:
            self.known_faces = self._load_known_faces()
            print(f"Reloaded {len(self.known_faces)} known faces from disk")

    def _save_known_faces(self):
        """Save known face embeddings to disk."""
        with open(self.known_faces_file, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def get_face_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract face embedding from image using MTCNN."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None

        try:
            faces = self.mtcnn(image)
            if faces is not None and len(faces) > 0:
                face = faces[0].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.facenet(face)
                return embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"MTCNN face detection error: {e}")

        return None

    def register_face(self, name: str, image_bytes: bytes, display_name: str = None) -> bool:
        """Register a new face for recognition with fallback detection."""
        if not FACE_RECOGNITION_AVAILABLE:
            return False

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # Preprocess image
            width, height = image.size
            max_size = 1024
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Try MTCNN first
            embedding = self.get_face_embedding(image)

            # If MTCNN fails, try using YOLO face detector
            if embedding is None:
                print("MTCNN failed, trying YOLO face detection...")
                face_results = self.face_model(image, conf=0.3, verbose=False)[0]

                if len(face_results.boxes) > 0:
                    box = face_results.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    padding = 30
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image.width, x2 + padding)
                    y2 = min(image.height, y2 + padding)
                    face_crop = image.crop((x1, y1, x2, y2))
                    embedding = self.get_face_embedding(face_crop)

            if embedding is None:
                print(f"Failed to detect face in image for {name}")
                return False

            self.known_faces[name] = {
                "embedding": embedding,
                "display_name": display_name or name
            }
            self._save_known_faces()
            print(f"Successfully registered face for {name} ({display_name or name})")
            return True

        except Exception as e:
            print(f"Error registering face: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_known_faces(self) -> list:
        """Return list of registered face names."""
        return list(self.known_faces.keys())

    def identify_face(self, face_image: Image.Image, threshold: float = 0.8) -> Optional[Tuple[str, str]]:
        """Identify a face from known faces."""
        if not FACE_RECOGNITION_AVAILABLE or not self.known_faces:
            return None

        embedding = self.get_face_embedding(face_image)

        if embedding is None:
            try:
                face_results = self.face_model(face_image, conf=0.2, verbose=False)[0]
                if len(face_results.boxes) > 0:
                    box = face_results.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(face_image.width, x2 + padding)
                    y2 = min(face_image.height, y2 + padding)
                    face_crop = face_image.crop((x1, y1, x2, y2))
                    embedding = self.get_face_embedding(face_crop)
            except Exception as e:
                print(f"Fallback face detection error: {e}")

        if embedding is None:
            return None

        best_match = None
        best_match_id = None
        best_distance = float('inf')

        for employee_id, face_data in self.known_faces.items():
            if isinstance(face_data, dict):
                known_embedding = face_data["embedding"]
                display_name = face_data.get("display_name", employee_id)
            else:
                known_embedding = face_data
                display_name = employee_id

            distance = np.linalg.norm(embedding - known_embedding)
            if distance < best_distance:
                best_distance = distance
                best_match_id = employee_id
                best_match = display_name

        print(f"Best match: {best_match} ({best_match_id}), distance: {best_distance:.3f}, threshold: {threshold}")

        if best_distance < threshold:
            return (best_match_id, best_match)
        return None

    def _normalize_label(self, label: str) -> str:
        """Normalize label names from Roboflow model."""
        # Map Roboflow labels to standardized format
        label_lower = label.lower().strip()

        # Positive PPE detections
        if label_lower in ["helmet", "hardhat", "hard hat", "hard-hat"]:
            return "Helmet"
        if label_lower in ["vest", "safety vest", "safety-vest", "hi-vis", "high-vis"]:
            return "Vest"
        if label_lower in ["goggles", "safety goggles", "glasses", "safety glasses"]:
            return "Goggles"
        if label_lower in ["mask", "face mask", "dust mask", "respirator"]:
            return "Mask"
        if label_lower in ["shoes", "safety shoes", "safety shoe", "boots", "safety boots"]:
            return "Safety Shoes"

        # Missing PPE violations
        if label_lower in ["no helmet", "no-helmet", "no_helmet", "missing helmet", "no hardhat", "no-hardhat"]:
            return "NO Helmet"
        if label_lower in ["no vest", "no-vest", "no_vest", "missing vest", "no safety vest"]:
            return "NO Vest"
        if label_lower in ["no goggles", "no-goggles", "no_goggles", "missing goggles"]:
            return "NO Goggles"
        if label_lower in ["no mask", "no-mask", "no_mask", "missing mask"]:
            return "NO Mask"
        if label_lower in ["no shoes", "no-shoes", "no_shoes", "no safety shoes", "missing shoes"]:
            return "NO Safety Shoes"

        # Person detection (not a PPE item)
        if label_lower in ["person", "human", "worker"]:
            return "Person"

        return label

    def _is_violation(self, label: str) -> bool:
        """Check if label indicates missing PPE."""
        return label.startswith("NO ")

    def _run_ppe_detection(self, image: Image.Image) -> list:
        """Run PPE detection using local YOLOv8 models."""
        detections = []
        
        # Run first model
        if self.ppe_model1:
            results = self.ppe_model1(image, conf=0.25, verbose=False)[0]
            if len(results.boxes) > 0:
                print(f"DEBUG [Detector]: Model 1 found {len(results.boxes)} objects. Names: {results.names}")
                print(f"DEBUG [Detector]: Raw labels found: {[results.names[int(b.cls[0])] for b in results.boxes]}")
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label_idx = int(box.cls[0])
                label = results.names[label_idx]
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': conf,
                    'model': 'model1'
                })
                
        # Run second model
        if self.ppe_model2:
            results = self.ppe_model2(image, conf=0.25, verbose=False)[0]
            if len(results.boxes) > 0:
                print(f"DEBUG [Detector]: Model 2 found {len(results.boxes)} objects. Names: {results.names}")
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label_idx = int(box.cls[0])
                label = results.names[label_idx]
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': conf,
                    'model': 'model2'
                })
        
        return detections

    def process_image(self, 
                      image: Image.Image, 
                      run_ppe: bool = True, 
                      run_face: bool = True,
                      cached_detections: Optional[dict] = None) -> tuple[Image.Image, dict]:
        """
        Process image and return annotated image with detections.
        Supports partial detection (PPE/Face only) to save CPU.
        """
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                # Convert BGR (OpenCV) to RGB (PIL)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                from io import BytesIO
                image = Image.open(BytesIO(image)).convert("RGB")
            
        draw = ImageDraw.Draw(image)

        try:
            # Try to load a reasonable font based on OS
            if os.name == 'nt':  # Windows
                font_path = "arial.ttf"
            else:  # macOS / Linux
                font_path = "/System/Library/Fonts/Helvetica.ttc"
                
            font = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 14)
        except:
            try:
                # Absolute fallback for Windows standard location
                if os.name == 'nt':
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                    font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
                else:
                    font = ImageFont.load_default()
                    font_small = font
            except:
                font = ImageFont.load_default()
                font_small = font

        # IMPORTANT: Preserve cached results if we are skipping this cycle
        detections = cached_detections if cached_detections else {
            "ppe": [],
            "faces": [],
            "violations": []
        }

        # --- PPE DETECTION ---
        if run_ppe:
            raw_ppe_detections = self._run_ppe_detection(image)
            # Clear old results when running a new cycle
            detections["ppe"] = []
            detections["violations"] = []
            
            for det in raw_ppe_detections:
                x1, y1, x2, y2 = det['bbox']
                raw_label = det['label']
                conf = det['confidence']
                label = self._normalize_label(raw_label)

                # Skip Person/Goggles as they might clutter the UI or aren't standard violations
                if label in ["Person", "Goggles"]: continue

                is_violation = self._is_violation(label)
                detection_info = {
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "is_violation": is_violation
                }
                detections["ppe"].append(detection_info)
                if is_violation: detections["violations"].append(detection_info)
            
            if len(detections["ppe"]) > 0:
                print(f"DEBUG [Detector]: PPE Processing result: {len(detections['ppe'])} items")

        # --- FACE DETECTION & RECOGNITION ---
        if run_face:
            detections["faces"] = []
            face_results = self.face_model(image, conf=0.3, verbose=False)[0]
            for box in face_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # Crop and identify with better padding for live feed
                padding = 50 
                face_crop = image.crop((max(0, x1-padding), max(0, y1-padding), 
                                      min(image.width, x2+padding), min(image.height, y2+padding)))
                
                # Using a more robust threshold for live recognition (1.05 instead of 0.85)
                identification = self.identify_face(face_crop, threshold=1.05)
                
                person_id, person_name = identification if identification else (None, None)
                detections["faces"].append({
                    "employee_id": person_id,
                    "name": person_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        # --- ANNOTATION (Always draw boxes using current detections) ---
        self._annotate_image(draw, detections, font, font_small)

        # Generate summary
        detections["summary"] = self._generate_summary(detections)
        
        # Add a simplified view for basic UI icons (requested by user)
        summary = detections["summary"]
        detections["simplified"] = {
            "helmet": summary["ppe_detected"].get("Helmet", 0) > 0,
            "vest": summary["ppe_detected"].get("Vest", 0) > 0,
            "shoes": summary["ppe_detected"].get("Safety Shoes", 0) > 0,
            "face": summary["faces_detected"] > 0
        }
        
        print(f"DEBUG [Detector]: PPE={len(detections['ppe'])}, Faces={len(detections['faces'])}, Simplified={detections['simplified']}")
        
        return image, detections

    def _annotate_image(self, draw, detections, font, font_small):
        """Draw detection boxes and labels on the image."""
        # Draw PPE
        for det in detections["ppe"]:
            x1, y1, x2, y2 = det["bbox"]
            label, is_violation = det["label"], det["is_violation"]
            color = self.colors.get(label, (0, 255, 0) if not is_violation else (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            text = f"{label} {det['confidence']:.2f}"
            text_bbox = draw.textbbox((x1, y1 - 2), text, font=font_small)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 2), text, fill=(255,255,255) if is_violation else (0,0,0), font=font_small)

        # Draw Faces
        for face in detections["faces"]:
            x1, y1, x2, y2 = face["bbox"]
            color = self.colors["face"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            name = face["name"] if face["name"] else f"Unknown {face['confidence']:.2f}"
            text = f"{name} ({face['employee_id']})" if face["employee_id"] else name
            text_bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 20), text, fill=(0,0,0), font=font)

    def _generate_summary(self, detections: dict) -> dict:
        """Generate a summary of safety compliance."""
        ppe_items = detections["ppe"]
        faces = detections["faces"]
        violations = detections["violations"]

        # Count PPE by type
        ppe_counts = {}
        for item in ppe_items:
            label = item["label"]
            if not item["is_violation"]:
                ppe_counts[label] = ppe_counts.get(label, 0) + 1

        violation_counts = {}
        for item in violations:
            label = item["label"]
            violation_counts[label] = violation_counts.get(label, 0) + 1

        identified = [f["employee_id"] for f in faces if f.get("employee_id")]
        identified_names = [f["name"] for f in faces if f.get("name")]

        return {
            "ppe_detected": ppe_counts,
            "violations": violation_counts,
            "total_ppe_items": len([p for p in ppe_items if not p["is_violation"]]),
            "total_violations": len(violations),
            "faces_detected": len(faces),
            "identified_persons": identified,
            "identified_names": [f["name"] for f in faces if f.get("name")],
            "safety_compliant": len(violations) == 0
        }
