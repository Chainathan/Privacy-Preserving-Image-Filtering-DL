import json
import os
import shutil
import random
from tqdm import tqdm
from collections import defaultdict

# --- CONFIG ---
COCO_JSON_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\original\screen_coco_segmentation.json'
IMAGES_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\original\images'
OUTPUT_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\balanced'
SPLIT_RATIO = 0.65  # 65% train, 35% val
DUPLICATIONS = 2  # Duplicate train set 2x
SEED = 42

TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, 'images/train')
VAL_IMG_DIR = os.path.join(OUTPUT_DIR, 'images/val')
ANN_DIR = os.path.join(OUTPUT_DIR, 'annotations')
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

# --- Load COCO ---
with open(COCO_JSON_PATH, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# --- Build lookup tables ---
image_id_to_image = {img['id']: img for img in images}
image_id_to_annotations = defaultdict(list)
for ann in annotations:
    image_id_to_annotations[ann['image_id']].append(ann)

# --- Stratified split (randomized) ---
random.seed(SEED)
image_ids = list(image_id_to_image.keys())
random.shuffle(image_ids)
split_index = int(len(image_ids) * SPLIT_RATIO)
train_ids = set(image_ids[:split_index])
val_ids = set(image_ids[split_index:])

# --- Prepare output containers ---
train_images = []
val_images = []
train_annotations = []
val_annotations = []
next_image_id = 1
next_annotation_id = 1

def copy_image_and_annotations(image, anns, dst_dir, copy_suffix=None):
    global next_image_id, next_annotation_id
    new_file_name = image['file_name']
    if copy_suffix:
        name, ext = os.path.splitext(image['file_name'])
        new_file_name = f"{name}_{copy_suffix}{ext}"

    # Copy image file
    src_path = os.path.join(IMAGES_DIR, image['file_name'])
    dst_path = os.path.join(dst_dir, new_file_name)
    shutil.copy2(src_path, dst_path)

    # Add new image entry
    new_image = image.copy()
    new_image['file_name'] = new_file_name
    new_image['id'] = next_image_id
    image_id_new = next_image_id
    next_image_id += 1

    new_anns = []
    for ann in anns:
        new_ann = ann.copy()
        new_ann['id'] = next_annotation_id
        new_ann['image_id'] = image_id_new
        next_annotation_id += 1
        new_anns.append(new_ann)

    return new_image, new_anns

# --- Handle train split and duplications ---
for img_id in tqdm(train_ids, desc='Processing train'):
    image = image_id_to_image[img_id]
    anns = image_id_to_annotations[img_id]

    # original + n duplicates
    for i in range(DUPLICATIONS + 1):
        suffix = None if i == 0 else f"copy{i}"
        new_img, new_anns = copy_image_and_annotations(image, anns, TRAIN_IMG_DIR, copy_suffix=suffix)
        train_images.append(new_img)
        train_annotations.extend(new_anns)

# --- Handle val split (no duplication) ---
for img_id in tqdm(val_ids, desc='Processing val'):
    image = image_id_to_image[img_id]
    anns = image_id_to_annotations[img_id]
    new_img, new_anns = copy_image_and_annotations(image, anns, VAL_IMG_DIR)
    val_images.append(new_img)
    val_annotations.extend(new_anns)

# --- Save JSONs ---
train_coco = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}
val_coco = {
    "images": val_images,
    "annotations": val_annotations,
    "categories": categories
}

with open(os.path.join(ANN_DIR, 'train.json'), 'w') as f:
    json.dump(train_coco, f, indent=2)

with open(os.path.join(ANN_DIR, 'val.json'), 'w') as f:
    json.dump(val_coco, f, indent=2)

print("Dataset split and duplication completed.")
