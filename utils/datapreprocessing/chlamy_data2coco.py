import os
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm

import datetime
import json
from utils.datapreprocessing.pycococreatortools import *

import sys

from src.globals import *

DATASET_NAME = "acdc_dataset"

INFO = {
    "description": "cellpose data in COCO format",
    "url": "",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "Florian Bridges",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [{"id": 1, "name": "", "url": ""}]

CATEGORIES = [
    {
        "id": 0,
        "name": "G1_or_G0",
        "supercategory": "Cell",
    },
    {
        "id": 1,
        "name": "SM_2_cell",
        "supercategory": "Cell",
    },
    {
        "id": 2,
        "name": "SM_4_cell",
        "supercategory": "Cell",
    },
    {
        "id": 3,
        "name": "SM_8_cell",
        "supercategory": "Cell",
    },
    {
        "id": 4,
        "name": "transition_into_2_cell",
        "supercategory": "Cell",
    },
    {
        "id": 5,
        "name": "transition_into_4_cell",
        "supercategory": "Cell",
    },
    {
        "id": 6,
        "name": "transition_into_8_cell",
        "supercategory": "Cell",
    },
    {
        "id": 7,
        "name": "Uncharacterized",
        "supercategory": "Cell",
    },
]


class Data2cocoConverter:
    def __init__(self) -> None:
        self.create_dir_to_path(path=os.getenv("DATA_PATH"), dir_name="chlamy_dataset")
        self.coco_data_path = os.path.join(os.getenv("DATA_PATH"), "chlamy_dataset")
        self.create_dir_to_path(path=os.getenv("DATA_PATH"), dir_name="chlamy_dataset")

    def create_dir_to_path(self, path, dir_name):
        if not os.path.exists(os.path.join(path, dir_name)):
            os.mkdir(os.path.join(path, dir_name))

    def convert(self):
        print("converting acdc dataset to coco format ...")

        raw_images_path = os.path.join(
            os.getenv("DATA_PATH"), "raw_data", "chlamy_dataset"
        )
        data_map = self.build_data_map(raw_images_path)

        index = data_map.index.to_numpy()
        np.random.seed(1221)
        np.random.shuffle(index)
        data_dict = {}
        data_dict["train"] = index[:21]
        data_dict["test"] = index[21:]

        print(data_map)
        print(
            f"test dataset:\n{data_map.iloc[data_dict['test']]['paths'].str.split('chlamy_dataset/').str[-1]}"
        )
        print(
            f"train dataset:\n{data_map.iloc[data_dict['train']]['paths'].str.split('chlamy_dataset/').str[-1]}"
        )
        for split_type in ["test", "train"]:
            self.create_dir_to_path(
                path=os.path.join(self.coco_data_path), dir_name=split_type
            )

            self.create_dir_to_path(
                path=os.path.join(self.coco_data_path, split_type), dir_name="images"
            )

            images_coco_data_path = os.path.join(
                self.coco_data_path, split_type, "images"
            )

            coco_output = self.acdc_to_json(
                data_map=data_map,
                index=data_dict[split_type],
                raw_images_path=raw_images_path,
                images_coco_data_path=images_coco_data_path,
            )

            with open(
                "{}/cell_acdc_coco_ds.json".format(
                    os.path.join(self.coco_data_path, split_type)
                ),
                "w",
            ) as output_json_file:
                json.dump(coco_output, output_json_file)

    def build_data_map(self, raw_images_path):
        print(raw_images_path)
        paths = []
        phase_contr_tifs = []
        phase_contr_npzs = []
        segms = []
        max_ids = []
        output_csvs = []

        for root, dirs, files in os.walk(raw_images_path):
            phase_contr_npz = ""
            phase_contr_tif = ""
            segm = ""
            max_id = -1
            output_csv = ""

            if root.find("Images") != -1:
                for filename in files:
                    if filename.find("PMT_T1.tif") != -1:
                        phase_contr_tif = filename
                    if filename.find("segm_cellpose70.npz") != -1:
                        segm = filename

                    if filename.find("output_cellpose70.csv") != -1:
                        output_csv = os.path.join(root, filename)

            paths.append(root)

            phase_contr_tifs.append(phase_contr_tif)
            segms.append(segm)
            output_csvs.append(output_csv)

        df = pd.DataFrame(
            data={
                "paths": paths,
                "phc_tif": phase_contr_tifs,
                "segm": segms,
                "output_csv": output_csvs,
            }
        )
        df_clean = df[df["segm"] != ""].copy().reset_index()

        return df_clean.sort_values(by=["paths"]).reset_index(drop=True)

    def process_acdc_position(
        self,
        data,
        base_image_id,
        segmentation_id,
        images_coco_data_path,
        output_csv,
    ):
        images: np.array = data[0]
        masks: np.array = data[1]

        output_df = pd.read_csv(output_csv)

        images_json = []
        annotations_json = []

        max_image = 1e6
        num_images: int = min(images.shape[0], masks.shape[0], max_image)

        for i in range(num_images):
            if (masks[i] > 0).sum():
                if "trasnsition_into_8_cell" in output_df.columns:
                    output_df["transition_into_8_cell"] = output_df[
                        "trasnsition_into_8_cell"
                    ]

                if "Uncharcterized" in output_df.columns:
                    output_df["Uncharacterized"] = output_df["Uncharcterized"]

                try:
                    df_out = output_df[
                        ["Cell_ID"] + [cat["name"] for cat in CATEGORIES]
                    ]
                except:
                    for col in output_df.columns:
                        print(col)
                    raise ValueError()
                df_out = df_out.set_index("Cell_ID")
                annotation_dict = df_out.to_dict("index")

                image_id = str(base_image_id) + "_" + str(i)

                image = images[i]
                mask = masks.max(axis=0)

                plt.imsave(
                    os.path.join(images_coco_data_path, image_id + ".png"),
                    image.astype(float),
                    cmap="gray",
                )
                h, w = image.shape
                image_info = create_image_info(
                    image_id, os.path.basename(image_id + ".png"), (w, h)
                )

                images_json.append(image_info)

                new_annotations_json, segmentation_id = self.extract_annotations(
                    mask=mask,
                    image=image,
                    image_id=image_id,
                    segmentation_id=segmentation_id,
                    annotation_dict=annotation_dict,
                )
                annotations_json += new_annotations_json

        return images_json, annotations_json, segmentation_id

    def map_class_id(self, anno_rec):
        for cat in CATEGORIES:
            if anno_rec[cat["name"]] == 1:
                return cat["id"]

        return 7

    def extract_annotations(
        self, mask, image, image_id, segmentation_id, annotation_dict
    ):
        annotations_json = []

        # save masks independently
        labels = np.unique(mask)
        for label in labels[1:]:
            class_id = self.map_class_id(annotation_dict[label])

            category_info = {
                "id": class_id,
                "is_crowd": False,
            }

            binary_mask: np.uint8 = (mask == label).astype(np.uint8)
            h, w = image.shape
            annotation_info = create_annotation_info(
                segmentation_id,
                image_id,
                category_info,
                binary_mask,
                (w, h),
                tolerance=0,
            )

            if annotation_info is not None:
                annotations_json.append(annotation_info)
                segmentation_id = segmentation_id + 1

        return annotations_json, segmentation_id

    def load_video(self, path, phc_tif, segm):
        vid = io.imread(os.path.join(path, phc_tif))

        masks = np.load(os.path.join(path, segm))["arr_0"]
        data = np.array([vid, masks], dtype="object")

        return data

    def acdc_to_json(self, data_map, index, raw_images_path, images_coco_data_path):
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": [],
        }

        segmentation_id = 0

        for id in tqdm(index):
            row = data_map.iloc[id]

            data = self.load_video(
                path=row["paths"],
                phc_tif=row["phc_tif"],
                segm=row["segm"],
            )

            base_image_id = (
                row["paths"]
                .replace(raw_images_path + "/", "")
                .replace("/Images", "")
                .replace("Position", "pos")
                .replace("/", "_")
            )

            images, annotations, segmentation_id = self.process_acdc_position(
                data=data,
                base_image_id=base_image_id,
                segmentation_id=segmentation_id,
                images_coco_data_path=images_coco_data_path,
                output_csv=row["output_csv"],
            )

            coco_output["images"] += images
            coco_output["annotations"] += annotations

        return coco_output


if __name__ == "__main__":
    acdc_conv = Data2cocoConverter()
    acdc_conv.convert()
