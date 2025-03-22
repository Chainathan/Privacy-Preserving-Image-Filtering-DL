import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIG ---
CCPD_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\CCPD2019.tar\CCPD2019\ccpd_base'
OUTPUT_JSON = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\CCPD2019.tar\CCPD2019\ccpd_coco_segmentation.json'
SAM2_CHECKPOINT = r"C:\Users\Sai\Documents\Neu\Masters_Project\sam2\checkpoints\sam2.1_hiera_large.pt"
MODEL_CFG = r"C:\Users\Sai\Documents\Neu\Masters_Project\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
DEVICE = 'cuda'
CATEGORY_ID = 1
CATEGORY_NAME = 'license_plate'

# --- Init SAM2 ---
sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# --- Output Containers ---
images_coco = []
annotations_coco = []
categories_coco = [{"id": CATEGORY_ID, "name": CATEGORY_NAME}]

image_id = 1
annotation_id = 1

files = [f for f in os.listdir(CCPD_DIR) if f.endswith(('.jpg', '.png'))]
files = files[:15000]
for filename in tqdm(files):
    img_path = os.path.join(CCPD_DIR, filename)

    # Parse corner field from filename
    fields = filename.split('-')
    if len(fields) < 3:
        print(f"Skipping malformed filename: {filename}")
        continue

    try:
        bbox_pts = fields[2].split('_')
        x1, y1 = map(int, bbox_pts[0].split('&'))
        x2, y2 = map(int, bbox_pts[1].split('&'))
        xyxy_box = [x1, y1, x2, y2]
    except Exception as e:
        print(f"Error parsing bbox from: {filename} | {e}")
        continue

    # Load image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    h_img, w_img = img_np.shape[:2]

    # Add image entry
    images_coco.append({
        "id": image_id,
        "file_name": filename,
        "height": h_img,
        "width": w_img
    })

    # Run SAM2
    predictor.set_image_batch([img_np])
    masks_batch, scores_batch, _ = predictor.predict_batch(
        None, None,
        box_batch=[np.array([xyxy_box], dtype=np.float32)],
        multimask_output=True
    )

    best_mask = masks_batch[0][np.argmax(scores_batch[0])]

    if best_mask.sum() == 0:
        print(f"Empty mask for {filename}")
        continue

    rle = maskUtils.encode(np.asfortranarray(best_mask.astype(np.uint8)))
    area = float(maskUtils.area(rle))
    bbox = maskUtils.toBbox(rle).tolist()

    annotations_coco.append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": CATEGORY_ID,
        "segmentation": {
            "size": rle['size'],
            "counts": rle['counts'].decode('ascii')
        },
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    })

    image_id += 1
    annotation_id += 1

# --- Save COCO JSON ---
coco = {
    "images": images_coco,
    "annotations": annotations_coco,
    "categories": categories_coco
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(coco, f, indent=2)

print(f"COCO-format segmentation dataset saved to: {OUTPUT_JSON}")
