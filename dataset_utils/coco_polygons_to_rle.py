import json
from pycocotools import mask as maskUtils
import numpy as np

# --- CONFIG ---
INPUT_COCO_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\midv500_data\midv500_coco_segmentation.json'
OUTPUT_COCO_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\midv500_data\midv500_coco_rle_segmentation.json'

# --- Load COCO file ---
with open(INPUT_COCO_PATH, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories']

# --- Build mapping from image_id to image size ---
image_id_to_size = {img['id']: (img['height'], img['width']) for img in images}

# --- Convert each polygon to RLE ---
new_annotations = []

for ann in annotations:
    if isinstance(ann['segmentation'], list):  # Polygon format
        img_h, img_w = image_id_to_size[ann['image_id']]

        # Convert polygon(s) to RLE using pycocotools
        rles = maskUtils.frPyObjects(ann['segmentation'], img_h, img_w)

        # If multiple polygons, merge them
        rle = maskUtils.merge(rles)
        rle['counts'] = rle['counts'].decode('ascii')  # Make JSON serializable

        new_ann = ann.copy()
        new_ann['segmentation'] = rle
        new_ann['area'] = float(maskUtils.area(rle))  # Optional: recompute area
        new_ann['bbox'] = maskUtils.toBbox(rle).tolist()  # Optional: recompute bbox
        new_ann['iscrowd'] = ann.get('iscrowd', 0)  # Optional: set iscrowd flag

        new_annotations.append(new_ann)
    else:
        # Already in RLE format or unexpected format
        new_annotations.append(ann)

# --- Save new COCO JSON ---
rle_coco = {
    'images': images,
    'annotations': new_annotations,
    'categories': categories
}

with open(OUTPUT_COCO_PATH, 'w') as f:
    json.dump(rle_coco, f, indent=2)

print(f"✅ Converted polygon masks to RLE and saved to: {OUTPUT_COCO_PATH}")
