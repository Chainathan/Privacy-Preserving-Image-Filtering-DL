import os
import json
'''
project/
├── data/
│   ├── screens/
│   │   ├── images/
│   │   └── annotations.json
│   ├── plates/
│   │   ├── images/
│   │   └── annotations.json
│   └── id_cards/
│       ├── images/
│       └── annotations.json
├── merged_dataset.json
'''
BASE_PATH = r"C:\Users\Sai\Documents\Neu\Masters_Project\PerceptionPrivacy\datasets\classes"

CAT_NAME_1 = "license_plate"
JSON_PATH_1 = os.path.join(BASE_PATH, "CCPD2019", "ccpd_coco_segmentation.json")
IMAGE_FOLDER_1 = "CCPD2019/images"

CAT_NAME_2 = "screen"
JSON_PATH_2 = os.path.join(BASE_PATH, "Screen", "screen_coco_segmentation_balanced.json")
IMAGE_FOLDER_2 = "Screen/images_balanced"

CAT_NAME_3 = "id_card"
JSON_PATH_3 = os.path.join(BASE_PATH, "midv500_data", "midv500_coco_rle_segmentation.json")
IMAGE_FOLDER_3 = "midv500_data/images"

OUTPUT_JSON = os.path.join(BASE_PATH, "merged_dataset_coco_balanced.json")

datasets = [
    (CAT_NAME_1, JSON_PATH_1, IMAGE_FOLDER_1),
    (CAT_NAME_2, JSON_PATH_2, IMAGE_FOLDER_2),
    (CAT_NAME_3, JSON_PATH_3, IMAGE_FOLDER_3),
]

# COCO format merged output
merged = {
    "images": [],
    "annotations": [],
    "categories": []
}

image_id_offset = 0
annotation_id_offset = 0
category_id_map = {}

for new_cat_id, (cat_name, json_path, rel_img_folder) in enumerate(datasets, start=1):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Add new category
    merged["categories"].append({
        "id": new_cat_id,
        "name": cat_name,
        "supercategory": "sensitive_object"
    })
    category_id_map[cat_name] = new_cat_id

    # Map image IDs to avoid conflicts
    image_id_map = {}
    for img in data["images"]:
        old_id = img["id"]
        new_id = old_id + image_id_offset
        img["id"] = new_id
        # img["file_name"] = os.path.join(rel_img_folder, img["file_name"])
        img["file_name"] = f"{rel_img_folder}/{img['file_name']}"
        merged["images"].append(img)
        image_id_map[old_id] = new_id

    for ann in data["annotations"]:
        ann["id"] += annotation_id_offset
        ann["image_id"] = image_id_map[ann["image_id"]]
        ann["category_id"] = new_cat_id  # Assign new category
        merged["annotations"].append(ann)

    # Update offsets
    image_id_offset = max(img["id"] for img in data["images"]) + 1
    annotation_id_offset = max(ann["id"] for ann in data["annotations"]) + 1

# Save merged annotation file
with open(OUTPUT_JSON, "w") as f:
    json.dump(merged, f)

print(f"Merged dataset saved to: {OUTPUT_JSON}")