import sys, os, time, torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import PersonDetector

def benchmark_detector():
    print("Initializing detector for benchmark...")
    detector = PersonDetector()
    
    # Create a dummy image (640x360)
    img_array = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        detector.process_image(image)
    
    # Benchmark
    print("Running benchmark (30 iterations)...")
    start_time = time.time()
    for i in range(30):
        iter_start = time.time()
        detector.process_image(image)
        iter_end = time.time()
        if (i+1) % 5 == 0:
            print(f"Iteration {i+1}/30: {(iter_end - iter_start)*1000:.2f}ms")
            
    end_time = time.time()
    avg_time = (end_time - start_time) / 30
    print(f"\nAverage inference time: {avg_time*1000:.2f}ms")
    print(f"Projected FPS: {1/avg_time:.2f}")

if __name__ == "__main__":
    benchmark_detector()
