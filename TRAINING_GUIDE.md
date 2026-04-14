# PPE Model Training Guide

## Step 3: Creating Annotation Files

### Understanding YOLO Annotation Format

Each image needs a corresponding `.txt` file with the same name. For example:
- `images/train/worker1.jpg` → `labels/train/worker1.txt`
- `images/train/site_photo.png` → `labels/train/site_photo.txt`

### Annotation Format

Each line in the `.txt` file represents one object:
```
<class_id> <x_center> <y_center> <width> <height>
```

- **class_id**: The class number from your `data.yaml` (0-11)
- **x_center**: X coordinate of bounding box center (0.0 to 1.0, relative to image width)
- **y_center**: Y coordinate of bounding box center (0.0 to 1.0, relative to image height)
- **width**: Width of bounding box (0.0 to 1.0, relative to image width)
- **height**: Height of bounding box (0.0 to 1.0, relative to image height)

### Your Class IDs (from data.yaml)
```
0: helmet
1: no_helmet
2: vest
3: no_vest
4: gloves
5: no_gloves
6: boots
7: no_boots
8: safety_goggles
9: no_safety_goggles
10: ear_protection
11: no_ear_protection
```

### Example Annotation

If you have an image `worker1.jpg` (1920x1080 pixels) with:
- A helmet at pixel coordinates: top-left (500, 100), bottom-right (700, 250)
- A vest at pixel coordinates: top-left (450, 300), bottom-right (750, 600)

Calculate normalized values:
```
# Helmet (class 0)
x_center = (500 + 700) / 2 / 1920 = 0.3125
y_center = (100 + 250) / 2 / 1080 = 0.1620
width = (700 - 500) / 1920 = 0.1042
height = (250 - 100) / 1080 = 0.1389

# Vest (class 2)
x_center = (450 + 750) / 2 / 1920 = 0.3125
y_center = (300 + 600) / 2 / 1080 = 0.4167
width = (750 - 450) / 1920 = 0.1563
height = (600 - 300) / 1080 = 0.2778
```

The `worker1.txt` file would contain:
```
0 0.3125 0.1620 0.1042 0.1389
2 0.3125 0.4167 0.1563 0.2778
```

---

## Recommended: Use LabelImg for Annotation

Manual annotation is tedious. Use **LabelImg** instead:

### Install LabelImg
```bash
pip install labelImg
```

### Run LabelImg
```bash
labelImg /Users/souravdey/Projects/person-detector/dataset/images/train /Users/souravdey/Projects/person-detector/dataset/classes.txt /Users/souravdey/Projects/person-detector/dataset/labels/train
```

### LabelImg Workflow
1. Open LabelImg with the command above
2. Click "Change Save Dir" → select `dataset/labels/train`
3. Click "PascalVOC" button at left → change to "YOLO"
4. Open image folder → select `dataset/images/train`
5. For each image:
   - Press `W` to create a bounding box
   - Draw box around each PPE item
   - Select the class from dropdown
   - Press `Ctrl+S` to save
   - Press `D` to go to next image

---

## Step 4: Running Training

Once you have:
- Images in `dataset/images/train/` and `dataset/images/val/`
- Corresponding labels in `dataset/labels/train/` and `dataset/labels/val/`

Run the training script:
```bash
cd /Users/souravdey/Projects/person-detector
python train_ppe.py
```

---

## Dataset Tips

1. **Split your data**: Put ~80% images in `train/`, ~20% in `val/`
2. **Minimum images**: At least 100-500 images per class for decent results
3. **Variety**: Include different lighting, angles, backgrounds
4. **Balance**: Try to have similar number of examples per class
5. **Quality**: Clear images where PPE is visible

---

## Troubleshooting

### "No labels found"
- Check that `.txt` files are in `labels/train/` not `images/train/`
- Ensure filenames match exactly (case-sensitive)

### "Class X not found"
- Verify class IDs in annotation files match `data.yaml`
- Class IDs start at 0, not 1

### Low accuracy
- Add more training images
- Ensure annotations are accurate (boxes tight around objects)
- Train for more epochs
