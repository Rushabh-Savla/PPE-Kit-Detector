"""
Detection module combining PPE detection, face detection, and face recognition.
Dual-model approach: Model 1 (medium) for helmets/shoes, Model 2 (nano) for vests/gloves.
"""
import pickle
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
    print("WARNING: facenet-pytorch not installed. Face recognition disabled.")


class PersonDetector:
    """Combined PPE and face detection/recognition."""

    def __init__(self, known_faces_dir: str = "../known_faces"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # ── PPE Model 1: keremberke/yolov8m-protective-equipment-detection ─────────
        # Classes: Hard-Hat, Gloves, Mask, Safety-Shoe, Safety Vest, Person +  NO- variants
        print("Loading PPE model 1 (medium)...")
        try:
            p = hf_hub_download(repo_id="keremberke/yolov8m-protective-equipment-detection", filename="best.pt")
            self.ppe_model1 = YOLO(p).to(self.device)
            print(f"  Model 1 labels: {list(self.ppe_model1.names.values())}")
        except Exception as e:
            print(f"  WARNING: Failed to load PPE model 1: {e}")
            self.ppe_model1 = None

        # ── PPE Model 2: Tanishjain9/yolov8n-ppe-detection-6classes ───────────────
        # Classes include Vest explicitly
        print("Loading PPE model 2 (nano)...")
        try:
            p = hf_hub_download(repo_id="Tanishjain9/yolov8n-ppe-detection-6classes", filename="best.pt")
            self.ppe_model2 = YOLO(p).to(self.device)
            print(f"  Model 2 labels: {list(self.ppe_model2.names.values())}")
        except Exception as e:
            print(f"  WARNING: Failed to load PPE model 2: {e}")
            self.ppe_model2 = None

        # ── Face detection model ───────────────────────────────────────────────────
        print("Loading face detection model...")
        try:
            p = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
            self.face_model = YOLO(p).to(self.device)
            print("  Face detection model loaded.")
        except Exception as e:
            print(f"  WARNING: Failed to load face model: {e}")
            self.face_model = None

        # ── Face Recognition ───────────────────────────────────────────────────────
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(exist_ok=True)
        self.known_faces_file = self.known_faces_dir / "embeddings.pkl"

        if FACE_RECOGNITION_AVAILABLE:
            print("Loading face recognition (FaceNet)...")
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                thresholds=[0.5, 0.6, 0.6],
                min_face_size=20,
                post_process=True
            )
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.known_faces = self._load_known_faces()
            print(f"  Loaded {len(self.known_faces)} known face(s).")
        else:
            self.mtcnn = None
            self.facenet = None
            self.known_faces = {}

        # ── PIL drawing colors ────────────────────────────────────────────────────
        self.colors = {
            "Helmet": (0, 200, 0),
            "Vest": (0, 255, 0),
            "Goggles": (0, 180, 80),
            "Mask": (0, 140, 100),
            "Safety Shoes": (0, 120, 80),
            "Gloves": (0, 160, 60),
            "NO Helmet": (255, 0, 0),
            "NO Vest": (255, 60, 60),
            "NO Goggles": (255, 50, 50),
            "NO Mask": (255, 100, 100),
            "NO Safety Shoes": (255, 120, 120),
            "NO Gloves": (255, 80, 80),
            "face": (255, 255, 0),
            "default": (128, 128, 128),
        }

        print("All models loaded successfully!")

    # ─────────────────────────── FACE REGISTRATION / IDENTITY ───────────────────

    def _load_known_faces(self) -> dict:
        if self.known_faces_file.exists():
            with open(self.known_faces_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def reload_faces(self):
        if FACE_RECOGNITION_AVAILABLE:
            self.known_faces = self._load_known_faces()
            print(f"Reloaded {len(self.known_faces)} known face(s).")

    def _save_known_faces(self):
        with open(self.known_faces_file, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def get_known_faces(self) -> list:
        return list(self.known_faces.keys())

    def get_face_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract face embedding with MTCNN."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
        try:
            faces = self.mtcnn(image)
            if faces is not None and len(faces) > 0:
                face = faces[0].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.facenet(face)
                return emb.cpu().numpy()[0]
        except Exception as e:
            print(f"MTCNN error: {e}")
        return None

    def register_face(self, name: str, image_bytes: bytes, display_name: str = None) -> bool:
        """Register a new face for recognition."""
        if not FACE_RECOGNITION_AVAILABLE:
            return False
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            # Resize if too large
            w, h = image.size
            if max(w, h) > 1024:
                ratio = 1024 / max(w, h)
                image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

            embedding = self.get_face_embedding(image)

            # Fallback: crop with YOLO face detector then retry
            if embedding is None and self.face_model:
                face_results = self.face_model(image, conf=0.3, verbose=False)[0]
                if len(face_results.boxes) > 0:
                    box = face_results.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    pad = 30
                    face_crop = image.crop((max(0, x1-pad), max(0, y1-pad),
                                           min(image.width, x2+pad), min(image.height, y2+pad)))
                    embedding = self.get_face_embedding(face_crop)

            if embedding is None:
                print(f"Failed to detect face for {name}")
                return False

            self.known_faces[name] = {
                "embedding": embedding,
                "display_name": display_name or name
            }
            self._save_known_faces()
            print(f"Registered face: {name} ({display_name or name})")
            return True
        except Exception as e:
            print(f"Error registering face: {e}")
            import traceback; traceback.print_exc()
            return False

    def identify_face(self, face_image: Image.Image, threshold: float = 1.05) -> Optional[Tuple[str, str]]:
        """Identify face against known faces. Returns (employee_id, display_name) or None."""
        if not FACE_RECOGNITION_AVAILABLE or not self.known_faces:
            return None

        embedding = self.get_face_embedding(face_image)

        # Fallback: crop with YOLO
        if embedding is None and self.face_model:
            try:
                res = self.face_model(face_image, conf=0.2, verbose=False)[0]
                if len(res.boxes) > 0:
                    box = res.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    pad = 20
                    crop = face_image.crop((max(0, x1-pad), max(0, y1-pad),
                                           min(face_image.width, x2+pad), min(face_image.height, y2+pad)))
                    embedding = self.get_face_embedding(crop)
            except Exception as e:
                print(f"Fallback face detect error: {e}")

        if embedding is None:
            return None

        best_id, best_name, best_dist = None, None, float('inf')
        for emp_id, face_data in self.known_faces.items():
            known_emb = face_data["embedding"] if isinstance(face_data, dict) else face_data
            disp = face_data.get("display_name", emp_id) if isinstance(face_data, dict) else emp_id
            dist = float(np.linalg.norm(embedding - known_emb))
            if dist < best_dist:
                best_dist, best_id, best_name = dist, emp_id, disp

        print(f"Face match: {best_name} dist={best_dist:.3f} threshold={threshold}")
        return (best_id, best_name) if best_dist < threshold else None

    # ─────────────────────────── LABEL NORMALIZATION ─────────────────────────────

    def _normalize_label(self, label: str) -> str:
        """Map raw model label → clean standardized label."""
        raw = label.lower().strip()
        # normalize underscores/dashes to spaces for matching
        raw_sp = raw.replace("_", " ").replace("-", " ")

        # ── Positive PPE ──────────────────────────────────────────────────────────
        if raw_sp in ("helmet", "hard hat", "hardhat", "hard-hat"):
            return "Helmet"
        if raw_sp in ("vest", "safety vest", "hi vis", "high vis", "hi-vis", "high-vis"):
            return "Vest"
        if raw_sp in ("goggles", "safety goggles", "glasses", "safety glasses"):
            return "Goggles"
        if raw_sp in ("mask", "face mask", "dust mask", "respirator"):
            return "Mask"
        if raw_sp in ("shoes", "safety shoe", "safety shoes", "boots", "safety boots"):
            return "Safety Shoes"
        if raw_sp in ("gloves", "glove", "safety gloves"):
            return "Gloves"

        # ── Violation / Missing PPE ───────────────────────────────────────────────
        # Handle "no helmet", "no-helmet", "no vest" etc.
        for prefix in ("no ", "without ", "missing "):
            if raw_sp.startswith(prefix):
                item = raw_sp[len(prefix):].strip()
                norm = self._normalize_label(item)
                if norm not in ("Person", item):      # avoid infinite recursion
                    return f"NO {norm}"

        # ── Person / Unknown ─────────────────────────────────────────────────────
        if raw_sp in ("person", "human", "worker", "people"):
            return "Person"

        return label   # return as-is if not recognized

    def _is_violation(self, label: str) -> bool:
        return label.startswith("NO ")

    # ─────────────────────────── PPE DETECTION ───────────────────────────────────

    def _run_ppe_detection(self, image: Image.Image) -> list:
        """Dual model pass: Model 1 (accuracy) + Model 2 (vest/gloves coverage)."""
        dets = []

        if self.ppe_model1:
            results = self.ppe_model1(image, conf=0.20, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self.ppe_model1.names[int(box.cls[0])]
                conf = float(box.conf[0])
                dets.append({"bbox": [x1, y1, x2, y2], "label": label, "confidence": conf, "model": "m1"})

        if self.ppe_model2:
            results = self.ppe_model2(image, conf=0.20, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self.ppe_model2.names[int(box.cls[0])]
                conf = float(box.conf[0])
                dets.append({"bbox": [x1, y1, x2, y2], "label": label, "confidence": conf, "model": "m2"})

        return dets

    # ─────────────────────────── MAIN PROCESS IMAGE ──────────────────────────────

    def process_image(self, image, run_ppe: bool = True, run_face: bool = True,
                      cached_detections: Optional[dict] = None) -> tuple:
        """
        Process an image (BGR numpy array OR PIL Image).
        Returns (annotated_pil_image, detections_dict).
        """
        # ── Convert to PIL RGB ────────────────────────────────────────────────────
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image)).convert("RGB")

        draw = ImageDraw.Draw(image)

        # ── Load font ─────────────────────────────────────────────────────────────
        try:
            fp = "arial.ttf" if os.name == "nt" else "/System/Library/Fonts/Helvetica.ttc"
            font = ImageFont.truetype(fp, 16)
            font_sm = ImageFont.truetype(fp, 14)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
            except:
                font = ImageFont.load_default()
                font_sm = font

        # ── Build detections dict ─────────────────────────────────────────────────
        detections = cached_detections if cached_detections else {
            "ppe": [], "faces": [], "violations": []
        }

        # ── PPE Detection ─────────────────────────────────────────────────────────
        if run_ppe:
            raw = self._run_ppe_detection(image)
            detections["ppe"] = []
            detections["violations"] = []

            for det in raw:
                label = self._normalize_label(det["label"])
                if label == "Person":
                    continue          # skip bare person detections
                is_viol = self._is_violation(label)
                info = {
                    "label": label,
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "is_violation": is_viol,
                }
                detections["ppe"].append(info)
                if is_viol:
                    detections["violations"].append(info)

        # ── Face Detection & Recognition ──────────────────────────────────────────
        if run_face and self.face_model:
            detections["faces"] = []
            face_res = self.face_model(image, conf=0.25, verbose=False)[0]

            for box in face_res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Crop with padding for better recognition
                pad = 50
                face_crop = image.crop((
                    max(0, x1 - pad), max(0, y1 - pad),
                    min(image.width, x2 + pad), min(image.height, y2 + pad)
                ))

                identification = self.identify_face(face_crop, threshold=1.05)
                person_id, person_name = identification if identification else (None, None)

                detections["faces"].append({
                    "employee_id": person_id,
                    "name": person_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

        # ── Draw boxes on PIL image ───────────────────────────────────────────────
        self._annotate_image(draw, detections, font, font_sm)

        # ── Generate summary ──────────────────────────────────────────────────────
        summary = self._generate_summary(detections)
        detections["summary"] = summary

        # ── Simplified status for dashboard icons ────────────────────────────────
        ppe_det = summary.get("ppe_detected", {})
        detections["simplified"] = {
            "helmet": ppe_det.get("Helmet", 0) > 0,
            "vest":   ppe_det.get("Vest", 0) > 0,
            "shoes":  ppe_det.get("Safety Shoes", 0) > 0,
            "goggles":ppe_det.get("Goggles", 0) > 0,
            "gloves": ppe_det.get("Gloves", 0) > 0,
            "face":   summary.get("faces_detected", 0) > 0,
        }

        return image, detections

    # ─────────────────────────── ANNOTATION ──────────────────────────────────────

    def _annotate_image(self, draw, detections, font, font_sm):
        """Draw bounding boxes + labels on PIL draw context."""
        for det in detections.get("ppe", []):
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            is_viol = det["is_violation"]
            color = self.colors.get(label, (0, 255, 0) if not is_viol else (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{label} {det['confidence']:.2f}"
            tb = draw.textbbox((x1, y1 - 2), text, font=font_sm)
            draw.rectangle(tb, fill=color)
            draw.text((x1, y1 - 2), text, fill=(255, 255, 255) if is_viol else (0, 0, 0), font=font_sm)

        for face in detections.get("faces", []):
            x1, y1, x2, y2 = face["bbox"]
            color = self.colors["face"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            name = face["name"] if face["name"] else f"Unknown {face['confidence']:.2f}"
            text = f"{name} ({face['employee_id']})" if face["employee_id"] else name
            tb = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((x1, y1 - 20), text, fill=(0, 0, 0), font=font)

    # ─────────────────────────── SUMMARY ─────────────────────────────────────────

    def _generate_summary(self, detections: dict) -> dict:
        ppe_items = detections.get("ppe", [])
        faces = detections.get("faces", [])
        violations = detections.get("violations", [])

        ppe_counts = {}
        for item in ppe_items:
            if not item["is_violation"]:
                ppe_counts[item["label"]] = ppe_counts.get(item["label"], 0) + 1

        viol_counts = {}
        for item in violations:
            viol_counts[item["label"]] = viol_counts.get(item["label"], 0) + 1

        return {
            "ppe_detected": ppe_counts,
            "violations": viol_counts,
            "total_ppe_items": sum(ppe_counts.values()),
            "total_violations": len(violations),
            "faces_detected": len(faces),
            "identified_persons": [f["employee_id"] for f in faces if f.get("employee_id")],
            "identified_names": [f["name"] for f in faces if f.get("name")],
            "safety_compliant": len(violations) == 0,
        }
