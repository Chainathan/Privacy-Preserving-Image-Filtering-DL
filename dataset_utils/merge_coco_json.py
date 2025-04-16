import json
import os

# --- CONFIG ---
COCO1_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Credit_Cards.v1i.coco\creditcard_test_segmentation.json'
COCO2_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Credit_Cards.v1i.coco\creditcard_valid_segmentation.json'
OUT_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Credit_Cards.v1i.coco\creditcard_val_merged.json'

# --- Load both datasets ---
with open(COCO1_PATH, 'r') as f1, open(COCO2_PATH, 'r') as f2:
    coco1 = json.load(f1)
    coco2 = json.load(f2)

# --- Verify categories match ---
if coco1['categories'] != coco2['categories']:
    raise ValueError("Category mismatch between the two datasets!")

categories = coco1['categories']
merged_images = []
merged_annotations = []

image_id_map = {}  # old_id -> new_id
next_image_id = 1
next_ann_id = 1

# --- Merge images from coco1 ---
for img in coco1['images']:
    new_img = img.copy()
    old_id = new_img['id']
    new_img['id'] = next_image_id
    image_id_map[old_id] = next_image_id
    merged_images.append(new_img)
    next_image_id += 1

# --- Merge annotations from coco1 ---
for ann in coco1['annotations']:
    new_ann = ann.copy()
    new_ann['id'] = next_ann_id
    new_ann['image_id'] = image_id_map[ann['image_id']]
    merged_annotations.append(new_ann)
    next_ann_id += 1

# --- Merge images from coco2 ---
for img in coco2['images']:
    new_img = img.copy()
    old_id = new_img['id']
    new_img['id'] = next_image_id
    image_id_map[old_id] = next_image_id
    merged_images.append(new_img)
    next_image_id += 1

# --- Merge annotations from coco2 ---
for ann in coco2['annotations']:
    new_ann = ann.copy()
    new_ann['id'] = next_ann_id
    new_ann['image_id'] = image_id_map[ann['image_id']]
    merged_annotations.append(new_ann)
    next_ann_id += 1

# --- Save merged file ---
merged_coco = {
    'images': merged_images,
    'annotations': merged_annotations,
    'categories': categories
}

with open(OUT_PATH, 'w') as f:
    json.dump(merged_coco, f, indent=2)

print(f"Merged dataset saved to {OUT_PATH}")
