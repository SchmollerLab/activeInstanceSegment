try:
    from globals import *
except:
    from src.globals import *

import json
import random as rd
import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def get_subset_dataset(sample_ids, dataset_name):
    return list(
        filter(
            lambda image: image["image_id"] in sample_ids,
            DatasetCatalog.get(dataset_name),
        )
    )


def build_register_function(image_ids, dataset_full):
    def register_function():
        return get_subset_dataset(image_ids, dataset_full)

    return register_function


def get_dataset_name(dataset, dsplit):

    if dsplit in DATASETS_DSPLITS[dataset]:
        return dataset + "_" + dsplit
    else:
        raise Exception("Dsplit `{}` not found in dataset `{}`".format(dsplit, dataset))


def register_datasets():

    for dataset in DATASETS_DSPLITS.keys():
        print("registering {} dataset".format(dataset))

        dsplits = DATASETS_DSPLITS[dataset]

        for dsplit in dsplits:
            if not get_dataset_name(dataset, dsplit) in MetadataCatalog:
                register_coco_instances(
                    get_dataset_name(dataset, dsplit),
                    {},
                    BASE_DATA_PATH + dataset + "/" + dsplit + "/" + REL_PATH_JSON,
                    BASE_DATA_PATH + dataset + "/" + dsplit + "/" + REL_PATH_IMAGES,
                )


def register_by_ids(dataset_name, image_ids, output_dir, dataset_full):

    if os.path.exists(output_dir + "/" + dataset_name + "_coco_format.json"):
        os.remove(output_dir + "/" + dataset_name + "_coco_format.json")

    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

    # if dataset_name in MetadataCatalog:
    #    raise Exception("Dataset name: '" + dataset_name + "is already taken")

    DatasetCatalog.register(
        dataset_name, build_register_function(image_ids, dataset_full)
    )
    MetadataCatalog.get(dataset_name).set(thing_classes=[CELL])

    return dataset_name


if __name__ == "__main__":
    register_datasets()
    get_dataset_name(ACDC_LARGE, "test_1")
