from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import json

def check_nano_model():
    print("--- PPE Nano Model ---")
    try:
        path = hf_hub_download(repo_id="Tanishjain9/yolov8n-ppe-detection-6classes", filename="best.pt")
        model = YOLO(path)
        print(json.dumps(model.names, indent=2))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_nano_model()
