from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch
import json

def debug_model():
    print("--- PPE Model ---")
    try:
        path = hf_hub_download(repo_id="keremberke/yolov8m-protective-equipment-detection", filename="best.pt")
        model = YOLO(path)
        print(json.dumps(model.names, indent=2))
        
        print("\n--- Face Model ---")
        face_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        face_model = YOLO(face_path)
        print(json.dumps(face_model.names, indent=2))
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    debug_model()
