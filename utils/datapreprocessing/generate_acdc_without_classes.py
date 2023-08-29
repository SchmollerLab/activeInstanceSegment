import os
import json

os.system("cp -r ./data/acdc_large/ ./data/acdc_large_cls/")

for dtype in ["train", "test"]:
    with open(f"data/acdc_large_cls/{dtype}/cell_acdc_coco_ds.json", "r") as file:
        coco_data = json.load(file)

    coco_data["categories"] = [{"id": 0, "name": "cell", "supercategory": "cell"}]
    annotations = coco_data["annotations"]
    coco_data["annotations"] = []

    for anno in annotations:
        anno["category_id"] = 0
        coco_data["annotations"].append(anno)

    with open(f"data/acdc_large/{dtype}/cell_acdc_coco_ds.json", "w") as file:
        json.dump(coco_data, file)
