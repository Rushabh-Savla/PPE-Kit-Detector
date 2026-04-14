import sys, os, cv2, torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import PersonDetector

def test_detection_logic():
    print("Initializing detector for logic test...")
    detector = PersonDetector()
    
    # Create a dummy image
    img_array = np.zeros((360, 640, 3), dtype=np.uint8)
    # Add a white square to simulate an object
    img_array[100:200, 100:200] = 255
    image = Image.fromarray(img_array)
    
    print("Processing dummy image...")
    # Mocking _run_ppe_detection to return some test boxes
    original_ppe = detector._run_ppe_detection
    
    def mocked_ppe(img):
        return [
            {'bbox': [100, 100, 200, 200], 'label': 'helmet', 'confidence': 0.95, 'model': 'test'},
            {'bbox': [300, 100, 400, 200], 'label': 'Vest', 'confidence': 0.88, 'model': 'test'}
        ]
    
    detector._run_ppe_detection = mocked_ppe
    
    # Mock face model results
    original_face = detector.face_model
    class MockResult:
        def __init__(self):
            self.boxes = [type('Box', (), {'xyxy': [torch.tensor([50, 50, 150, 150])], 'conf': [torch.tensor(0.99)]})]
            self.names = {0: 'FACE'}
            
    detector.face_model = lambda img, **kwargs: [MockResult()]
    
    # Process
    res_img, detections = detector.process_image(image)
    
    print("--- Detections ---")
    import json
    # Use a safe serializer for non-JSON objects if any (there shouldn't be anymore)
    print(json.dumps(detections, indent=2, default=str))
    
    # Save the result to check visually if possible (though I can't see it, I can check if it exists)
    res_img.save("backend/test_result_rendering.png")
    print("Result image saved to backend/test_result_rendering.png")

if __name__ == "__main__":
    test_detection_logic()
