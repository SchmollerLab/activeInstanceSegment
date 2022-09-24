import os
import math
import numpy as np
from PIL import Image
from skimage import exposure, io
import matplotlib.pyplot as plt

import datetime
import json
import re
import fnmatch
import pycococreatortools

from typing import Callable


BASE_PATH = "../../data/TimeLapse_2D"
DATA_SAVE_PATH = "../../data/processed_data/"
ROOT_DIR = "../../data/dataInCOCO/"
IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"
TEST = "test"
TRAIN = "train"

INFO = {
    "description": "Cell ACDC data in COCO format",
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


def iterate_images(function):

    id: int = 0
    split_type: str = TEST
    directory = os.fsencode(BASE_PATH)
    for obj in os.listdir(directory):
        name = os.fsdecode(obj)
        if name.find("_labeled") != -1:
            for pos in os.listdir(os.fsencode(BASE_PATH + "/" + name)):
                pos_name = os.fsdecode(pos)
                for file in os.listdir(
                    os.fsencode(BASE_PATH + "/" + name + "/" + pos_name + "/Images")
                ):
                    filename = os.fsdecode(file)
                    if filename.find("_phase_contr.tif") != -1:
                        base_filename = filename.replace("phase_contr.tif", "")
                data = load_video(name, pos_name, base_filename)
                function(data, str(id), split_type)
                id += 1
                split_type = TRAIN


def store_image(data: np.array, id: int, split_type: str = "train") -> None:

    images: np.array = data[0]
    masks: np.array = data[1]

    num_images: int = images.shape[0]

    for i in range(num_images):
        full_id = id + "_" + str(i)

        # save image as png
        plt.imsave(
            os.path.join(ROOT_DIR, split_type, "images", full_id + ".png"),
            images[i],
            cmap="gray",
        )

        # save masks independently
        mask_full = masks[i]

        labels = np.unique(mask_full)
        for label in labels[1:]:
            mask: np.uint8 = (mask_full == label).astype(np.uint8)
            plt.imsave(
                os.path.join(
                    ROOT_DIR,
                    split_type,
                    "annotations",
                    full_id + "_cell_" + str(label) + ".png",
                ),
                mask,
                cmap="gray",
            )


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
    file_name_prefix = basename_no_extension + ".*"
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

                    print(annotation_filename)
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
                        tolerance=2,
                    )

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open("{}/cell_acdc_coco_ds.json".format(path_dir), "w") as output_json_file:
        json.dump(coco_output, output_json_file)


def main():
    print("running main ...")
    # prepare images
    iterate_images(store_image)

    image_id = 1
    segmentation_id = 1

    for split_type in [TEST, TRAIN]:
        convert_data_to_coco(
            image_id, segmentation_id, os.path.join(ROOT_DIR, split_type)
        )


if __name__ == "__main__":
    main()
