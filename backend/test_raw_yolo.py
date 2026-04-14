import sys, os, cv2, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    path = hf_hub_download(repo_id="keremberke/yolov8m-protective-equipment-detection", filename="best.pt")
    model = YOLO(path).to(device)
    print("Model loaded")
    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"Frame captured: {frame.shape}")
        # Run raw YOLO on BGR frame
        results = model(frame, conf=0.1)[0] # Lowered conf for debug
        print(f"Boxes found: {len(results.boxes)}")
        for b in results.boxes:
            print(f"  Class: {results.names[int(b.cls[0])]} Conf: {float(b.conf[0]):.4f}")
    else:
        print("Failed to capture frame")
except Exception as e:
    print(f"Error: {e}")
