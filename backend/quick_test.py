import sys
import os
from PIL import Image
import numpy as np
import cv2

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import PersonDetector

def test_single_inference():
    detector = PersonDetector()
    if not detector.ppe_model:
        print("Error: PPE model not loaded")
        return

    print(f"Model Names: {detector.ppe_model.names}")
    
    # Create a dummy image or load one if available
    dummy_img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    
    _, results = detector.process_image(dummy_img)
    print(f"Simplified results: {results.get('simplified')}")
    print(f"PPE Detections: {results.get('ppe')}")

if __name__ == "__main__":
    test_single_inference()
