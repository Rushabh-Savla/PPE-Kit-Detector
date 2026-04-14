from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch

def debug_model():
    print("Loading PPE model...")
    try:
        path = hf_hub_download(repo_id="keremberke/yolov8m-protective-equipment-detection", filename="best.pt")
        model = YOLO(path)
        print("Model Names:", model.names)
        
        face_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        face_model = YOLO(face_path)
        print("Face Model Names:", face_model.names)
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    debug_model()
