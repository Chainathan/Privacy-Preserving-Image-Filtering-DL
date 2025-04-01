import os
import json
import random
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as maskUtils
from collections import defaultdict

# --- CONFIG ---
COCO_JSON = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\merged_dataset_coco_balanced.json'
IMAGES_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes'
OUTPUT_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\yolo_lp_screen_id'

# COCO_JSON = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\screen_coco_segmentation.json'
# IMAGES_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\images'
# OUTPUT_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen_yolo'
SPLIT_RATIO = 0.8  # Train/Val split
SEED = 42

# --- Create YOLOv8 directory structure ---
images_train_dir = os.path.join(OUTPUT_DIR, 'images/train')
images_val_dir = os.path.join(OUTPUT_DIR, 'images/val')
labels_train_dir = os.path.join(OUTPUT_DIR, 'labels/train')
labels_val_dir = os.path.join(OUTPUT_DIR, 'labels/val')

os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)

# --- Load COCO ---
with open(COCO_JSON, 'r') as f:
    coco = json.load(f)

image_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
image_ids = list(image_id_to_file.keys())

# --- Split into train and val ---
# --- Stratified Split by Class ---
class_to_image_ids = defaultdict(set)

# Step 1: Build mappings
for ann in coco['annotations']:
    image_id = ann['image_id']
    class_id = ann['category_id']
    class_to_image_ids[class_id].add(image_id)

train_ids = set()
val_ids = set()
seen = set()

# Step 2: Stratified split for each class
for class_id, image_ids in class_to_image_ids.items():
    image_ids = list(image_ids)
    random.seed(SEED)
    random.shuffle(image_ids)

    split_idx = int(len(image_ids) * SPLIT_RATIO)
    class_train = image_ids[:split_idx]
    class_val = image_ids[split_idx:]

    for img_id in class_train:
        if img_id not in seen:
            train_ids.add(img_id)
            seen.add(img_id)

    for img_id in class_val:
        if img_id not in seen:
            val_ids.add(img_id)
            seen.add(img_id)

# Sanity check
print(f"Stratified Split: {len(train_ids)} train images, {len(val_ids)} val images")

# --- Group annotations by image_id ---
image_to_anns = {}
for ann in coco['annotations']:
    image_to_anns.setdefault(ann['image_id'], []).append(ann)

# --- Process all images ---
for image_id, anns in tqdm(image_to_anns.items()):
    file_name = image_id_to_file[image_id]
    width, height = image_id_to_size[image_id]
    image_path = os.path.join(IMAGES_DIR, file_name)

    lines = []

    for ann in anns:
        seg = ann['segmentation']
        if not isinstance(seg, dict) or 'counts' not in seg:
            continue

        rle = seg.copy()
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
        mask = maskUtils.decode(rle)

        # Extract contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim != 2 or len(contour) < 3:
                continue

            # Normalize points
            norm_coords = []
            for x, y in contour:
                norm_x = x / width
                norm_y = y / height
                norm_coords.extend([round(norm_x, 6), round(norm_y, 6)])

            class_id = ann['category_id'] - 1  # COCO is 1-indexed, YOLO is 0-indexed
            line = f"{class_id} " + " ".join(map(str, norm_coords))
            lines.append(line)

    if not lines:
        continue  # Skip images with no valid masks

    file_name = os.path.basename(file_name)
    # Decide if train or val
    if image_id in train_ids:
        img_dst = os.path.join(images_train_dir, file_name)
        label_dst = os.path.join(labels_train_dir, os.path.splitext(file_name)[0] + '.txt')
    else:
        img_dst = os.path.join(images_val_dir, file_name)
        label_dst = os.path.join(labels_val_dir, os.path.splitext(file_name)[0] + '.txt')

    # Copy image
    shutil.copy2(image_path, img_dst)

    # Save label
    with open(label_dst, 'w') as f:
        f.write("\n".join(lines))

# --- Write data.yaml ---
class_names = [cat['name'] for cat in coco['categories']]
num_classes = len(class_names)

data_yaml = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val

nc: {num_classes}
names: {class_names}
"""

with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
    f.write(data_yaml.strip())

print("YOLOv8 dataset conversion complete.")
