import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools import mask as maskUtils

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIG ---
COCO_JSON_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen Detection YOLOv8\dataset\screen.json'
IMAGES_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen Detection YOLOv8\dataset\images'  # Path to where the image files are stored
NEW_COCO_JSON_PATH = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\Screen Detection YOLOv8\dataset\screen_segmentation.json'
SAM2_CHECKPOINT = r"C:\Users\Sai\Documents\Neu\Masters_Project\sam2\checkpoints\sam2.1_hiera_large.pt"
MODEL_CFG = r"C:\Users\Sai\Documents\Neu\Masters_Project\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
DEVICE = 'cuda'  # or 'cpu'
MULTIMASK = True  # Set to False if you only want a single mask per box

# --- Load COCO JSON ---
with open(COCO_JSON_PATH, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

# images = images[:10]  # Modify if needed

# --- Build SAM2 ---
sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# --- Prepare Image-to-Annotation Map ---
image_id_to_annotations = {}
for ann in annotations:
    image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)

# --- Conversion ---
new_annotations = []
next_ann_id = 1

batch_size = 1  # Modify if needed
for i in tqdm(range(0, len(images), batch_size)):
    batch_imgs = []
    batch_boxes = []
    img_metas = []

    for j in range(i, min(i + batch_size, len(images))):
        img_info = images[j]
        img_path = os.path.join(IMAGES_DIR, img_info['file_name'])
        img = np.array(Image.open(img_path).convert('RGB'))
        batch_imgs.append(img)
        img_metas.append(img_info)

        # Get and convert boxes
        anns = image_id_to_annotations.get(img_info['id'], [])
        xyxy_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            xyxy_boxes.append([x, y, x + w, y + h])

        if xyxy_boxes:  # only append if not empty
            batch_boxes.append(np.array(xyxy_boxes, dtype=np.float32))
        else:
            batch_boxes.append(np.empty((0, 4), dtype=np.float32))

    # Skip batch if no boxes
    if all(len(b) == 0 for b in batch_boxes):
        continue

    predictor.set_image_batch(batch_imgs)
    masks_batch, scores_batch, _ = predictor.predict_batch(
        None, None,
        box_batch=batch_boxes,
        multimask_output=MULTIMASK
    )

    for img_info, anns, masks, scores, boxes in zip(img_metas, batch_boxes, masks_batch, scores_batch, batch_boxes):
        if len(masks) == 0:
            print(f"No masks for image {img_info['id']}")
            continue

        if boxes.shape[0] == 1:
            masks = masks[None, ...]
            scores = scores[None, ...]

        selected_masks = masks[range(len(masks)), np.argmax(scores, axis=-1)] if MULTIMASK else masks[:, 0]

        for mask, ann, box in zip(selected_masks, image_id_to_annotations[img_info['id']], boxes):
            if mask.sum() == 0:
                print(f"Empty mask for image {img_info['id']}, annotation {ann['id']}") 

            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            area = float(maskUtils.area(rle))
            bbox = maskUtils.toBbox(rle).tolist()

            new_ann = {
                "id": next_ann_id,
                "image_id": ann['image_id'],
                "category_id": ann['category_id'],
                "segmentation": {
                    "size": rle['size'],
                    "counts": rle['counts'].decode('ascii')
                },
                "area": area,
                "bbox": bbox,
                "iscrowd": ann.get('iscrowd', 0),
            }
            new_annotations.append(new_ann)
            next_ann_id += 1

# --- Create New COCO Dataset ---
new_coco = {
    "images": images,
    "annotations": new_annotations,
    "categories": categories
}

with open(NEW_COCO_JSON_PATH, 'w') as f:
    json.dump(new_coco, f, indent=2)

print(f"Saved converted dataset to: {NEW_COCO_JSON_PATH}")