"""
PPE Detection Model Training Script
===================================
This script fine-tunes a YOLOv8 model on your custom PPE dataset.

Usage:
    python train_ppe.py

Requirements:
    - Images in dataset/images/train/ and dataset/images/val/
    - Labels in dataset/labels/train/ and dataset/labels/val/
    - data.yaml configured with correct path and classes
"""

from ultralytics import YOLO
import os

# Configuration
DATASET_PATH = "/Users/souravdey/Projects/person-detector/dataset/data.yaml"
OUTPUT_DIR = "/Users/souravdey/Projects/person-detector/models"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16  # Reduce to 8 if you run out of GPU memory
PATIENCE = 20    # Early stopping patience


def check_dataset():
    """Verify dataset structure before training."""
    base_path = "/Users/souravdey/Projects/person-detector/dataset"

    train_images = os.path.join(base_path, "images", "train")
    val_images = os.path.join(base_path, "images", "val")
    train_labels = os.path.join(base_path, "labels", "train")
    val_labels = os.path.join(base_path, "labels", "val")

    # Check directories exist
    for path, name in [(train_images, "Training images"),
                       (val_images, "Validation images"),
                       (train_labels, "Training labels"),
                       (val_labels, "Validation labels")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} directory not found: {path}")
            return False

    # Count files
    train_img_count = len([f for f in os.listdir(train_images)
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_img_count = len([f for f in os.listdir(val_images)
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    train_label_count = len([f for f in os.listdir(train_labels)
                             if f.endswith('.txt')])
    val_label_count = len([f for f in os.listdir(val_labels)
                           if f.endswith('.txt')])

    print("=" * 50)
    print("Dataset Summary:")
    print("=" * 50)
    print(f"Training images:   {train_img_count}")
    print(f"Training labels:   {train_label_count}")
    print(f"Validation images: {val_img_count}")
    print(f"Validation labels: {val_label_count}")
    print("=" * 50)

    if train_img_count == 0:
        print("ERROR: No training images found!")
        print("Add .jpg/.png images to: dataset/images/train/")
        return False

    if train_label_count == 0:
        print("ERROR: No training labels found!")
        print("Use LabelImg to create annotations in: dataset/labels/train/")
        return False

    if train_img_count != train_label_count:
        print(f"WARNING: Image count ({train_img_count}) != Label count ({train_label_count})")
        print("Each image should have a corresponding .txt label file")

    return True


def train():
    """Run the training process."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify dataset
    if not check_dataset():
        print("\nPlease fix the dataset issues above before training.")
        return

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Early stopping patience: {PATIENCE}")
    print()

    # Load pre-trained YOLOv8 model (medium size, good balance of speed/accuracy)
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium),
    #          yolov8l.pt (large), yolov8x.pt (extra large)
    model = YOLO("yolov8m.pt")

    # Train the model
    results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        save=True,
        project=OUTPUT_DIR,
        name="ppe_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,  # Generate training plots
    )

    # The best model is automatically saved at:
    # models/ppe_detector/weights/best.pt
    best_model_path = os.path.join(OUTPUT_DIR, "ppe_detector", "weights", "best.pt")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best model saved to: {best_model_path}")
    print("\nTo use this model, update backend/detector.py:")
    print(f'    self.ppe_model1 = YOLO("{best_model_path}")')
    print("=" * 50)

    return results


if __name__ == "__main__":
    train()
