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
# Dataset list: (new_category_name, path_to_json, relative_image_folder)
datasets = [
    ("screen", "data/screens/annotations.json", "screens/images"),
    ("license_plate", "data/plates/annotations.json", "plates/images"),
    ("id_card", "data/id_cards/annotations.json", "id_cards/images"),
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
        img["file_name"] = os.path.join(rel_img_folder, img["file_name"])
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
output_path = "merged_dataset.json"
with open(output_path, "w") as f:
    json.dump(merged, f)

print(f"Merged dataset saved to: {output_path}")
