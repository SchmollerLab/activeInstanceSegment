import os
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm

import datetime
import json
import pycococreatortools

import sys

sys.path.insert(0, sys.path[0] + "/..")
from globals import *

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
        "name": "cell",
        "supercategory": "cell",
    }
]


class Data2cocoConverter:
    def __init__(self) -> None:
        self.create_dir_to_path(path=os.getenv("DATA_PATH"), dir_name="acdc_large")
        self.coco_data_path = os.path.join(os.getenv("DATA_PATH"), "acdc_large")
        self.create_dir_to_path(
                path=os.getenv("DATA_PATH"), dir_name="acdc_large"
            )

    def create_dir_to_path(self, path, dir_name):
        if not os.path.exists(os.path.join(path, dir_name)):
            os.mkdir(os.path.join(path, dir_name))

    def convert(self):

        print("converting acdc dataset to coco format ...")

        
        raw_images_path = os.path.join(os.getenv("DATA_PATH"), "raw_data", "acdc_large")
        data_map = self.build_data_map(raw_images_path)
    
        index = data_map.index.to_numpy()
        np.random.seed(1221)
        np.random.shuffle(index)
        data_dict = {}
        data_dict["train"] = index[:33]
        data_dict["test"] = index[33:]

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

        paths = []
        phase_contr_tifs = []
        phase_contr_npzs = []
        segms = []
        max_ids = []

        base_dict = os.fsencode(raw_images_path)
        for acdc_ds in os.listdir(base_dict):
            acdc_ds_name = os.fsdecode(acdc_ds)
            if acdc_ds_name.find(".zip") == -1 and acdc_ds_name.find(".csv") == -1:
                experiment_dict = os.fsencode(
                    os.path.join(raw_images_path, acdc_ds_name)
                )
                for experiment in os.listdir(experiment_dict):
                    experiment_name = os.fsdecode(experiment)
                    for position in os.listdir(
                        os.fsencode(
                            raw_images_path + "/" + acdc_ds_name + "/" + experiment_name
                        )
                    ):
                        position_name = os.fsdecode(position)

                        phase_contr_npz = ""
                        phase_contr_tif = ""
                        segm = ""
                        max_id = -1

                        for file in os.listdir(
                            os.fsencode(
                                raw_images_path
                                + "/"
                                + acdc_ds_name
                                + "/"
                                + experiment_name
                                + "/"
                                + position_name
                                + "/Images"
                            )
                        ):
                            filename = os.fsdecode(file)
                            if (filename.find("Ph3_aligned.np") != -1) or (
                                filename.find("phase_contr_aligned.np") != -1
                            ):
                                phase_contr_npz = filename
                            if (filename.find("phase_contr.tif") != -1) or (
                                filename.find("Ph3.tif") != -1
                            ):
                                phase_contr_tif = filename
                            if filename.find("segm.npz") != -1:
                                segm = filename

                            if filename.find("output.csv") != -1:
                                output_df = pd.read_csv(
                                    raw_images_path
                                    + "/"
                                    + acdc_ds_name
                                    + "/"
                                    + experiment_name
                                    + "/"
                                    + position_name
                                    + "/Images/"
                                    + filename
                                )
                                max_id = max(output_df["frame_i"].values)

                        paths.append(
                            raw_images_path
                            + "/"
                            + acdc_ds_name
                            + "/"
                            + experiment_name
                            + "/"
                            + position_name
                            + "/Images"
                        )
                        phase_contr_npzs.append(phase_contr_npz)
                        phase_contr_tifs.append(phase_contr_tif)
                        segms.append(segm)
                        max_ids.append(max_id)

        df = pd.DataFrame(
            data={
                "paths": paths,
                "phc_npz": phase_contr_npzs,
                "phc_tif": phase_contr_tifs,
                "segm": segms,
                "max_image": max_ids,
            }
        )
        df_clean = df[df["segm"] != ""].copy().reset_index()
        df_clean["min_image"] = 0

        return df_clean

    def process_acdc_position(
        self, data, base_image_id, max_image, segmentation_id, images_coco_data_path
    ):

        images: np.array = data[0]
        masks: np.array = data[1]

        images_json = []
        annotations_json = []

        if max_image < 0:
            max_image = 1e6
        num_images: int = min(images.shape[0], masks.shape[0], max_image)

        for i in range(num_images):
            if (masks[i] > 0).sum():

                image_id = str(base_image_id) + "_" + str(i)

                image = images[i]
                mask = masks[i]

                plt.imsave(
                    os.path.join(images_coco_data_path, image_id + ".png"),
                    image.astype(float),
                    cmap="gray",
                )
                h, w = image.shape
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_id + ".png"), (w, h)
                )

                images_json.append(image_info)

                new_annotations_json, segmentation_id = self.extract_annotations(
                    mask=mask,
                    image=image,
                    image_id=image_id,
                    segmentation_id=segmentation_id,
                )
                annotations_json += new_annotations_json

        return images_json, annotations_json, segmentation_id

    def extract_annotations(self, mask, image, image_id, segmentation_id):

        annotations_json = []

        # save masks independently
        labels = np.unique(mask)
        for label in labels[1:]:

            class_id = 0
            category_info = {
                "id": class_id,
                "is_crowd": False,
            }

            binary_mask: np.uint8 = (mask == label).astype(np.uint8)
            h, w = image.shape
            annotation_info = pycococreatortools.create_annotation_info(
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

    def load_video(self, path, phc_npz, phc_tif, segm):

        if phc_npz != "" and isinstance(phc_npz, str):
            if phc_npz.find(".npz") != -1:
                vid = np.load(os.path.join(path, phc_npz))["arr_0"]
            else:
                vid = np.load(os.path.join(path, phc_npz))
        else:
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
                phc_npz=row["phc_npz"],
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
                max_image=row["max_image"],
                segmentation_id=segmentation_id,
                images_coco_data_path=images_coco_data_path,
            )

            coco_output["images"] += images
            coco_output["annotations"] += annotations

        return coco_output


if __name__ == "__main__":

    acdc_conv = Data2cocoConverter()
    acdc_conv.convert()
