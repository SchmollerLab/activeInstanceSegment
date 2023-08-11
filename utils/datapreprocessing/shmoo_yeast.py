import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd
from tqdm import tqdm

import datetime
import json
import re
import fnmatch
from utils.datapreprocessing.pycococreatortools import *

import sys

sys.path.insert(0, sys.path[0] + "/..")
from src.globals import *

IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"

INFO = {
    "description": "shmoo yeast data in COCO format",
    "url": "",
    "version": "0.1.0",
    "year": 2023,
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

TEST = "test"
TRAIN = "train"


class Data2cocoConverter:
    def __init__(self, root_dir, dataset_name, data_splits=[TEST, TRAIN]) -> None:
        self.dataset_name = dataset_name
        self.data_splits = data_splits
        self.raw_images_path = os.path.join(root_dir, "raw_data", dataset_name)
        self.save_images_path = os.path.join(root_dir, dataset_name)

        self.create_dir_to_path(root_dir, dataset_name)
        for data_split in data_splits:
            self.create_dir_to_path(self.save_images_path, data_split)
            self.create_dir_to_path(self.save_images_path, data_split + "/annotations")
            self.create_dir_to_path(self.save_images_path, data_split + "/images")

    def create_dir_to_path(self, path, dir_name):
        if not os.path.exists(os.path.join(path, dir_name)):
            os.mkdir(os.path.join(path, dir_name))

    def convert(self):
        print("converting {} images ...".format(self.dataset_name))

        # prepare images
        self.iterate_images()
        image_id = 1
        segmentation_id = 1

        for split_type in self.data_splits:
            print("converting to coco for split {}:".format(split_type))
            self.convert_data_to_coco(
                image_id,
                segmentation_id,
                os.path.join(self.save_images_path, split_type),
            )

    def iterate_images(self):
        pass


    def store_image(self, id, image, mask, split_type):
        # save image as png
        plt.imsave(
            os.path.join(self.save_images_path, split_type, "images", id + ".png"),
            image.astype(float),
            cmap="gray",
        )

        # save masks independently
        labels = np.unique(mask)
        for label in labels[1:]:
            mask_single_cell: np.uint8 = (mask == label).astype(np.uint8)
            plt.imsave(
                os.path.join(
                    self.save_images_path,
                    split_type,
                    "annotations",
                    id + "_cell_" + str(label) + ".png",
                ),
                mask_single_cell,
                cmap="gray",
            )


    def filter_for_png(self, root, files):
        file_types = ["*.png"]
        file_types = r"|".join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        return files

    def filter_for_annotations(self, root, files, image_filename):
        file_types = ["*.png"]
        file_types = r"|".join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + "_"
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [
            f
            for f in files
            if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])
        ]

        return files

    def convert_data_to_coco(self, image_id, segmentation_id, path_dir):
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": [],
        }

        for root, _, files in os.walk(os.path.join(path_dir, IMAGE_DIR_NAME)):
            image_files = self.filter_for_png(root, files)
            # go through each image
            for image_filename in tqdm(image_files):
                image = Image.open(image_filename)
                image_info = create_image_info(
                    image_id, os.path.basename(image_filename), image.size
                )
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                for root, _, files in os.walk(
                    os.path.join(path_dir, ANNOTATION_DIR_NAME)
                ):
                    annotation_files = self.filter_for_annotations(
                        root, files, image_filename
                    )

                    # go through each associated annotation
                    for annotation_filename in annotation_files:
                        class_id = [
                            x["id"]
                            for x in CATEGORIES
                            if x["name"] in annotation_filename
                        ][0]

                        category_info = {
                            "id": class_id,
                            "is_crowd": "crowd" in image_filename,
                        }
                        binary_mask = np.asarray(
                            Image.open(annotation_filename).convert("1")
                        ).astype(np.uint8)

                        annotation_info = create_annotation_info(
                            segmentation_id,
                            image_id,
                            category_info,
                            binary_mask,
                            image.size,
                            tolerance=0,
                        )

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

        with open(
            "{}/cell_acdc_coco_ds.json".format(path_dir), "w"
        ) as output_json_file:
            json.dump(coco_output, output_json_file)


class ShmooYeast2cocoConverter(Data2cocoConverter):
    def __init__(self, root_dir) -> None:
        dataset_name = "shmoo_yeast"
        super().__init__(root_dir, dataset_name)
        self.raw_images_path = os.path.join(self.raw_images_path, "data")

    def iterate_images(self):
        
        num_images = len(os.listdir(self.raw_images_path ))-1
        num_train = math.ceil(num_images * 0.8)

        ids = list(range(num_images))
        rd.shuffle(ids)
        ids_dict = {}
        ids_dict[TRAIN] = ids[:num_train]
        ids_dict[TEST] = ids[num_train:]

        for split_type in [TRAIN, TEST]:
            for id in tqdm(ids_dict[split_type]):

                npz = np.load(os.path.join(self.raw_images_path, str(id) + ".npz"))
    
                image = npz["image"]
                mask = npz["mask"]

                self.store_image(str(id), image, mask, split_type)



if __name__ == "__main__":
    cpc = ShmooYeast2cocoConverter(BASE_DATA_PATH)
    cpc.convert()
