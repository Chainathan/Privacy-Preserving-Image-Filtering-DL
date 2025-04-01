import json, os, shutil
from pathlib import Path

# --- CONFIG ---
src_json_path = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\screen_coco_segmentation.json'  # original COCO JSON
src_img_dir = Path(r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\images')           # root image folder (contains subdirs)
output_json_path = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\screen_coco_segmentation_balanced.json'
output_img_dir = Path(r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen\images_balanced')
oversample_class_id = 1                # category_id for screen
duplication_factor = 3                # how many copies to make (total = 1 original + 2 duplicates)

# --- PREP ---
os.makedirs(output_img_dir, exist_ok=True)

# Load original JSON
with open(src_json_path, 'r') as f:
    coco = json.load(f)

# Init new image + annotation lists
new_images = list(coco['images'])  # start with originals
new_annotations = list(coco['annotations'])

# Track IDs
next_image_id = max(img['id'] for img in new_images) + 1
next_ann_id = max(ann['id'] for ann in new_annotations) + 1

# Get screen image IDs
screen_image_ids = {
    ann['image_id']
    for ann in coco['annotations']
    if ann['category_id'] == oversample_class_id
}

# Get mapping: image_id → image metadata
image_id_map = {img['id']: img for img in coco['images']}

# Collect annotations per image
from collections import defaultdict
annotations_by_image = defaultdict(list)
for ann in coco['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

# --- DUPLICATION LOOP ---
for image_id in screen_image_ids:
    orig_img = image_id_map[image_id]
    orig_filename = orig_img['file_name']
    img_src_path = src_img_dir / orig_filename

    for i in range(1, duplication_factor):  # skip 0 (original already present)
        # Create new image file
        stem, ext = os.path.splitext(orig_filename)
        new_filename = f"{stem}_dup{i}{ext}"
        img_dst_path = output_img_dir / new_filename
        os.makedirs(img_dst_path.parent, exist_ok=True)
        shutil.copy(img_src_path, img_dst_path)

        # Add new image entry
        new_img = orig_img.copy()
        new_img['id'] = next_image_id
        new_img['file_name'] = str(img_dst_path.relative_to(output_img_dir))
        new_images.append(new_img)

        # Add new annotations
        for ann in annotations_by_image[image_id]:
            new_ann = ann.copy()
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = next_image_id
            new_annotations.append(new_ann)
            next_ann_id += 1

        next_image_id += 1

# Also copy original images
for img in coco['images']:
    src = src_img_dir / img['file_name']
    dst = output_img_dir / img['file_name']
    os.makedirs(dst.parent, exist_ok=True)
    if not dst.exists():
        shutil.copy(src, dst)

# Save updated JSON
balanced_coco = {
    'images': new_images,
    'annotations': new_annotations,
    'categories': coco['categories']
}
with open(output_json_path, 'w') as f:
    json.dump(balanced_coco, f)

print(f"Balanced dataset saved to {output_json_path}")
print(f"Images copied to {output_img_dir}")
