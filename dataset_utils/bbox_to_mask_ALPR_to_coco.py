import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIG ---
DATASET_DIR = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\License_plate_ALPR\UFPR-ALPR dataset'
OUT_JSON = r'C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes\License_plate_ALPR\UFPR-ALPR dataset\ALPR_segmentation.json'
SAM2_CHECKPOINT = r"C:\Users\Sai\Documents\Neu\Masters_Project\sam2\checkpoints\sam2.1_hiera_large.pt"
MODEL_CFG = r"C:\Users\Sai\Documents\Neu\Masters_Project\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
DEVICE = 'cuda'
CATEGORY_ID = 1
CATEGORY_NAME = "plate"

# --- Setup SAM2 ---
model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(model)

# --- Prepare Output Containers ---
images_coco = []
annotations_coco = []
categories_coco = [{"id": CATEGORY_ID, "name": CATEGORY_NAME}]

image_id = 1
annotation_id = 1

# --- Loop through dataset ---
files = []
for splitdir in os.listdir(DATASET_DIR):
    splitdir = os.path.join(DATASET_DIR, splitdir)
    if not os.path.isdir(splitdir):
        continue
    for trackdir in os.listdir(splitdir):
        trackdir = os.path.join(splitdir, trackdir)
        if not os.path.isdir(trackdir):
            continue
        count = 0
        for filename in os.listdir(trackdir):
            if filename.endswith(".txt"):
                files.append(os.path.join(trackdir, filename))
                count += 1
                if count >= 15:
                    break

for file in tqdm(files):
    txt_path = os.path.join(DATASET_DIR, file)
    img_path = os.path.splitext(txt_path)[0] + '.jpg'
    if not os.path.exists(img_path):
        img_path = os.path.splitext(txt_path)[0] + '.png'
        if not os.path.exists(img_path):
            continue

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    corner_line = [line for line in lines if line.startswith('corners:')]
    if not corner_line:
        continue

    # Parse corners
    corner_str = corner_line[0].split(':', 1)[1].strip()
    coords = [tuple(map(int, pt.split(','))) for pt in corner_str.split()]
    xs, ys = zip(*coords)
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    xyxy_box = [x_min, y_min, x_max, y_max]

    # Load image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    h_img, w_img = img_np.shape[:2]

    # Add image metadata
    image_entry = {
        "id": image_id,
        "file_name": os.path.basename(img_path),
        "height": h_img,
        "width": w_img
    }

    # Predict mask
    predictor.set_image_batch([img_np])
    masks_batch, scores_batch, _ = predictor.predict_batch(
        None, None,
        box_batch=[np.array([xyxy_box], dtype=np.float32)],
        multimask_output=True
    )

    best_mask = masks_batch[0][np.argmax(scores_batch[0])]

    if best_mask.sum() == 0:
        print(f"Warning: Empty mask for {img_path}")
        continue

    # Convert mask to RLE
    rle = maskUtils.encode(np.asfortranarray(best_mask.astype(np.uint8)))
    area = float(maskUtils.area(rle))
    bbox = maskUtils.toBbox(rle).tolist()

    annotation = {
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
    }

    images_coco.append(image_entry)
    annotations_coco.append(annotation)
    image_id += 1
    annotation_id += 1

# --- Save COCO JSON ---
coco_output = {
    "images": images_coco,
    "annotations": annotations_coco,
    "categories": categories_coco
}

with open(OUT_JSON, 'w') as f:
    json.dump(coco_output, f, indent=2)

print(f"COCO-format segmentation dataset saved to: {OUT_JSON}")
