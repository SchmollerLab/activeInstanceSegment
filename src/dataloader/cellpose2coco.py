import os
import math
import numpy as np
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
import pycococreatortools
 
from typing import Callable
 
BASE_PATH = "./data/raw_data/cellpose/data/"
DATA_SAVE_PATH = "./data/cellpose/"
#ROOT_DIR = "./data/"
IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"
TEST = "test"
TRAIN = "train"
 
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
 
def load_video(experiment_name, position, filename):
    path = BASE_PATH + "/" + experiment_name + "/" + position + "/Images/"
    print("loading:", path)
 
    vid = io.imread(path + filename + "phase_contr.tif")
    print("loaded video with shape:\t", vid.shape)
 
    lables = np.load(path + filename + "segm.npz")["arr_0"]
    print("loaded lables with shape:\t", lables.shape)
 
    data = np.array([vid, lables])
    print(data.shape)
 
    return data
 
def iterate_images():
 
    ids = list(range(540))
    print(ids[0])
    rd.shuffle(ids)
    print(ids[0])
    ids_dict = {}
    ids_dict[TRAIN] = ids[:431]
    ids_dict[TEST] = ids[431:]
 
    for split_type in [TRAIN,TEST]:
        for id in tqdm(ids_dict[split_type]):
            id_str = ("000" + str(id))[-3:]
            image = mpimg.imread(BASE_PATH + id_str + "_img.png")
            mask = mpimg.imread(BASE_PATH + id_str + "_masks.png")
            augment_store_image(str(id),image,mask, split_type)
 
def augment_store_image(full_id, image, mask, split_type):
 
    # flip_combs: [[horizontal,vertical]]
    if split_type == TRAIN:
        flip_combs = [[False,False],[True,False],[False,True],[True,True]] # start with only horizontal flip
    else:
        # no flip in test data
        flip_combs = [[False,False]]
   
    for flip_comb in flip_combs:
 
        full_id_with_flip = full_id + get_flip_str(flip_comb)
       
        image_flipped = flip_image(image, flip_comb)
        mask_full = flip_image(mask, flip_comb)
 
        # save masks independently
        store_image(full_id_with_flip, image_flipped, mask_full, split_type)
 
               
 
def store_image(id, image, mask, split_type):
 
    # save image as png
    plt.imsave(
        os.path.join(DATA_SAVE_PATH, split_type, "images", id + ".png"),
        image,
        cmap="gray",
    )
 
    # save masks independently
    labels = np.unique(mask)
    for label in labels[1:]:
        mask_single_cell: np.uint8 = (mask == label).astype(np.uint8)
        plt.imsave(
            os.path.join(
                DATA_SAVE_PATH,
                split_type,
                "annotations",
                id + "_cell_" + str(label) + ".png",
            ),
            mask_single_cell,
            cmap="gray",
        ) 
 
def flip_image(image, flip_type):
   
    for dim in [0,1]:
        if flip_type[dim]:
            image = np.flip(image,dim)
   
    return image
 
def get_flip_str(flip_type):
   
    flip_str = "H" + str(flip_type[0])[0] + "V" + str(flip_type[1])[0]
    return flip_str
   
            
            
        
def store_images(data: np.array, id: int, split_type: str = "train") -> None:
 
    images: np.array = data[0]
    masks: np.array = data[1]
 
    num_images: int = images.shape[0]
 
    for i in range(num_images):
       
        full_id = id + "_" + str(i)
        store_image(id,images[i], masks[i], split_type)
        
 
def filter_for_png(root, files):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
 
    return files
 
def filter_for_annotations(root, files, image_filename):
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
 
def convert_data_to_coco(image_id, segmentation_id, path_dir):
 
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }
 
    # filter for jpeg images
    for root, _, files in os.walk(os.path.join(path_dir, IMAGE_DIR_NAME)):
        image_files = filter_for_png(root, files)
        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size
            )
            coco_output["images"].append(image_info)
 
            # filter for associated png annotations
            for root, _, files in os.walk(os.path.join(path_dir, ANNOTATION_DIR_NAME)):
                annotation_files = filter_for_annotations(root, files, image_filename)
 
                   
                # go through each associated annotation
                for annotation_filename in annotation_files:
 
                    class_id = [
                        x["id"] for x in CATEGORIES if x["name"] in annotation_filename
                    ][0]
 
                    category_info = {
                        "id": class_id,
                        "is_crowd": "crowd" in image_filename,
                    }
                    binary_mask = np.asarray(
                        Image.open(annotation_filename).convert("1")
                    ).astype(np.uint8)
 
                    annotation_info = pycococreatortools.create_annotation_info(
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
           
    with open("{}/cell_acdc_coco_ds.json".format(path_dir), "w") as output_json_file:
        json.dump(coco_output, output_json_file)


def make_data_dirs(data_root_dir):
    os.mkdir(os.path.join(data_root_dir, "train"))
    os.mkdir(os.path.join(data_root_dir, "test"))
    os.mkdir(os.path.join(data_root_dir, "train","images"))
    os.mkdir(os.path.join(data_root_dir, "test","images"))
    os.mkdir(os.path.join(data_root_dir, "train","annotations"))
    os.mkdir(os.path.join(data_root_dir, "test","annotations"))
    
def main():
    print("running main ...")

    # mkdir test train
    make_data_dirs(DATA_SAVE_PATH)
    
    # prepare images
    iterate_images()
    image_id = 1
    segmentation_id = 1
 
    for split_type in [TEST, TRAIN]:
        convert_data_to_coco(
            image_id, segmentation_id, os.path.join(DATA_SAVE_PATH, split_type)
        )
 
if __name__ == "__main__":
    main()