import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

# --- CONFIG ---
COCO_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\merged_dataset_coco_balanced.json'
OUTPUT_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes'
TRAIN_JSON = os.path.join(OUTPUT_DIR, 'coco_bal_train.json')
VAL_JSON = os.path.join(OUTPUT_DIR, 'coco_bal_val.json')

SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load COCO ---
with open(COCO_PATH, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# --- Build mappings ---
image_id_to_image = {img['id']: img for img in images}
image_id_to_annotations = defaultdict(list)
image_id_to_classes = defaultdict(set)
class_to_image_ids = defaultdict(set)

for ann in annotations:
    image_id = ann['image_id']
    cat_id = ann['category_id']
    image_id_to_annotations[image_id].append(ann)
    image_id_to_classes[image_id].add(cat_id)
    class_to_image_ids[cat_id].add(image_id)

# --- Stratified split ---
train_ids = set()
val_ids = set()
seen = set()

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

# --- Save function ---
def create_coco_subset(image_ids, output_path):
    subset_images = [image_id_to_image[iid] for iid in image_ids]
    subset_annotations = [
        ann for iid in image_ids for ann in image_id_to_annotations[iid]
    ]
    subset_coco = {
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': categories
    }
    with open(output_path, 'w') as f:
        json.dump(subset_coco, f, indent=2)
    print(f"Saved {len(subset_images)} images to {output_path}")

# --- Write out train and val files ---
create_coco_subset(train_ids, TRAIN_JSON)
create_coco_subset(val_ids, VAL_JSON)
