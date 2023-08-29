import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, io

import matplotlib.image as mpimg
import random as rd
from tqdm import tqdm

import datetime
import json
import re
import fnmatch
from utils.datapreprocessing.pycococreatortools import *
from utils.datapreprocessing.data2coco import Data2cocoConverter

import sys

# shmoo_cells
from tiffile import imread
from cellpose import models

# to delete
import matplotlib.pyplot as plt
import skimage


sys.path.insert(0, sys.path[0] + "/..")
from src.globals import *


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
    {"id": 1, "name": "Budding", "supercategory": "cell"},
    {"id": 2, "name": "Arrested", "supercategory": "cell"},
    {"id": 3, "name": "Switching", "supercategory": "cell"},
    {"id": 4, "name": "Peanuts", "supercategory": "cell"},
    {"id": 5, "name": "Peanuts (short)", "supercategory": "cell"},
    {"id": 6, "name": "Peanuts (long)", "supercategory": "cell"},
    {"id": 7, "name": "Shmoo (single regular)", "supercategory": "cell"},
    {"id": 8, "name": "Shmoo (single long)", "supercategory": "cell"},
    {"id": 9, "name": "Shmoo (double regular)", "supercategory": "cell"},
    {"id": 10, "name": "Shmoo (multiple)", "supercategory": "cell"},
    {"id": 11, "name": "Shmoo (double, wide)", "supercategory": "cell"},
    {"id": 12, "name": "Shmoo (double, wide and short)", "supercategory": "cell"},
]

IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"

DATA_SPLIT_FRACTIONS = [
    {
        "type": "train",
        "fraction": 0.8,
    },
    {
        "type": "test",
        "fraction": 0.2,
    },
]

SEED = 1337


class ShmooYeast2cocoConverter(Data2cocoConverter):
    def __init__(self) -> None:
        dataset_name = "shmoo_yeast"
        super().__init__(
            dataset_name, info=INFO, licenses=LICENSES, categories=CATEGORIES
        )

        self.raw_images_path = os.path.join(
            DATA_PATH,
            "raw_data",
            dataset_name,
        )
        self.mask_dir = os.path.join(self.raw_images_path, "masks_by_ucid")
        self.image_dir = os.path.join(self.raw_images_path, "source_images")

        self.annotaion_df = pd.read_csv(
            os.path.join(self.mask_dir, "centroid_label_ucid_mapping.csv")
        )
        self.cellpose_model = models.Cellpose(gpu=True, model_type="cyto")

        self.minimum_cell_size = 50

    def get_data(self):
        data = []
        data_file_names = [
            file_name
            for file_name in os.listdir(self.image_dir)
            if (file_name.find(".tif") != -1 and file_name.find("BF_Position") != -1)
        ]

        print(f"loading data from {self.raw_images_path}")
        for data_file_name in tqdm(data_file_names):
            image_id = data_file_name.replace(".tif", "").replace("BF_Position", "")

            image, mask = self.load_annotated_image(image_id)
            image, mask, annotations = self.transform_data(image, mask)

            data.append(
                {
                    "image_id": image_id,
                    "image": image,
                    "mask": mask,
                    "annotations": annotations,
                }
            )

        return data

    def transform_data(self, image, mask):
        cellpose_mask, _, _, _ = self.cellpose_model.eval(
            [image], diameter=None, channels=[0, 0], flow_threshold=0.4, do_3D=False
        )
        cellpose_mask = cellpose_mask[0]

        new_id = 1
        new_mask = np.zeros(mask.shape)

        annotations = {}

        for cell_id in np.unique(mask):
            if cell_id == 0:  # 0 encodes background
                continue

            cellpose_ids = np.where(mask == cell_id, cellpose_mask, 0)
            unique_vals, counts = np.unique(cellpose_ids, return_counts=True)
            unique_vals_fil = np.unique(
                np.where(counts > self.minimum_cell_size, unique_vals, 0)
            )[1:]

            if (
                len(unique_vals_fil) == 1
                and len(self.annotaion_df[self.annotaion_df["ucid"] == cell_id]) > 0
            ):
                new_mask = np.where(
                    cellpose_mask == unique_vals_fil[0], new_id, new_mask
                )

                annotations[new_id] = self.annotaion_df[
                    self.annotaion_df["ucid"] == cell_id
                ]["tag.type"].values[0]
                new_id += 1

        background_avg_pixel_intensity = int(
            np.sum(np.where((mask == 0), image, 0)) / np.sum(mask == 0)
        )
        new_image = np.where(
            (cellpose_mask > 0) ^ (new_mask > 0), background_avg_pixel_intensity, image
        )

        return new_image, new_mask, annotations

    def load_tif_file(self, image_dir, file_name):
        return imread(os.path.join(image_dir, file_name))

    def load_annotated_image(self, id):
        image_file_name = f"BF_Position{id}.tif"
        mask_file_name = f"mask_BF_Position{id}.tif.out.tif"

        image = self.load_tif_file(self.image_dir, image_file_name)
        mask = self.load_tif_file(self.mask_dir, mask_file_name)

        return image, mask


if __name__ == "__main__":
    cpc = ShmooYeast2cocoConverter()
    cpc.main()
