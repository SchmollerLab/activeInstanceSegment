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
    "description": "",
    "url": "",
    "version": "0.1.0",
    "year": 2023,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [{"id": 1, "name": "", "url": ""}]

CATEGORIES = [
    {'id': 1, 'name': 'cell', 'supercategory': 'cell'},
]

IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"

DATA_SPLIT_FRACTIONS = [
    {"type": "train", "fraction": 0.8,},
    {"type": "test", "fraction": 0.2,},
]

SEED = 1337

class Data2cocoConverter:
    
    def __init__(self, dataset_name, data_split_fractions=DATA_SPLIT_FRACTIONS, seed=SEED, licenses=LICENSES, info=INFO, categories=CATEGORIES) -> None:
        self.dataset_name = dataset_name
        self.data_split_fractions = data_split_fractions
        self.seed = seed

        self.data_path = os.path.join(DATA_PATH, self.dataset_name)
        
        self.create_dir_to_path(
                path=DATA_PATH, dir_name=self.dataset_name
            )
       	
	self.licenses = licences
	self.info = info
	self.categories = categories
 
        for split_type in [ds["type"] for ds in self.data_split_fractions]:
            self.create_dir_to_path(
                path=os.path.join(self.data_path), dir_name=split_type
            )
            self.create_dir_to_path(
                    path=os.path.join(self.data_path,split_type), dir_name="images"
                )


    def get_data(self):
        #######################################################################################
        #
        # Format of data:
        # list of data points
        # a datapoint is a dict of the following
        # {
        #   "image_id": id of image,
        #   "image": image as numpy array,
        #   "mask": mask as numpy array with zero background and cell ids,
        #   "annotations": list of dicts [{'cell_id': 78, 'cell_class_id': 7}, ...]
        # }
        #
        #######################################################################################

        pass

    def create_dir_to_path(self, path, dir_name):
        if not os.path.exists(os.path.join(path, dir_name)):
            os.mkdir(os.path.join(path, dir_name))

    def split_data(self):             

        remaining_data = self.data
        len_data = len(self.data)
        # random shuffle
        rd.seed(self.seed)
        rd.shuffle(remaining_data)


        data_split = {}  
        for split_type in self.data_split_fractions:
            if len(remaining_data) == 0:
                print(f"data_split_fractions not specified correctly")
                break
            to_id = min(math.ceil(split_type["fraction"] * len_data), len(remaining_data))
            data_split[split_type["type"]] = remaining_data[: to_id]
            remaining_data = remaining_data[to_id :]
            print(split_type["type"], len(data_split[split_type["type"]]))

        return data_split

    def extract_annotations(self, mask, image, image_id, segmentation_id, annotation_dict):

        annotations_json = []

        # save masks independently
        cell_ids = np.unique(mask)
        for cell_id in cell_ids:

            if cell_id == 0:
                continue

            class_id = int(annotation_dict[cell_id])
            
            category_info = {
                "id": class_id,
                "is_crowd": False,
            }

            binary_mask: np.uint8 = (mask == cell_id).astype(np.uint8)
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
    
    def data_to_coco(self, data, split_type):
 
        coco_output = {
            "info": self.info,
            "licenses": self.licenses,
            "categories": self.categories,
            "images": [],
            "annotations": [],
        }

        segmentation_id = 0

        for data_point in tqdm(data):
            
            image_id = data_point["image_id"]
            image = data_point["image"]
            mask = data_point["mask"]
            annotations = data_point["annotations"]

            # save image
            plt.imsave(
                os.path.join(self.data_path, split_type, "images", image_id + ".png"),
                image.astype(float),
                cmap="gray",
            )

            h, w = image.shape

            image_info = create_image_info(
                image_id, os.path.basename(image_id + ".png"), (w, h)
            )

            coco_output["images"].append(image_info)

            new_annotations_json, segmentation_id = self.extract_annotations(
                mask=mask,
                image=image,
                image_id=image_id,
                segmentation_id=segmentation_id,
                annotation_dict = annotations,
            )
            coco_output["annotations"] += new_annotations_json

        for key in coco_output["annotations"][0].keys():
            print(key, type(coco_output["annotations"][0][key]))
        
        return coco_output
        

    def main(self):
        self.data = self.get_data()
        data_split = self.split_data()

        for split_type in data_split.keys():
            print(f"processing {split_type} dataset...")

            coco_output = self.data_to_coco(data_split[split_type], split_type)

            with open(
                "{}/cell_acdc_coco_ds.json".format(
                    os.path.join(self.data_path, split_type)
                ),
                "w",
            ) as output_json_file:
                json.dump(coco_output, output_json_file)
        
