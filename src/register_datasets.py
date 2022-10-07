try:
    from globals import *
except:
    from src.globals import *

import json
import random as rd
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def get_subset_dataset(sample_ids, dataset_name):
    return list(filter(lambda image: image["image_id"] in sample_ids, DatasetCatalog.get(dataset_name)))

def register_function_singe_point_dataset():
    
    with open(PATH_TRAIN_FULL_JSON) as file:
        train_dict = json.load(file)
        
    rd.seed(1337)
    images_meta_data = rd.sample(train_dict["images"], 5) 
    image_id = images_meta_data[0]["id"]
    return get_subset_dataset([image_id], TRAIN_DATASET_FULL)

def register_datasets():
    
    #train full
    if not TRAIN_DATASET_FULL in MetadataCatalog:
        register_coco_instances(
            TRAIN_DATASET_FULL,
            {},
            PATH_TRAIN_FULL_JSON,
            PATH_TRAIN_FULL_IMAGES,
        )
    
    # test full
    if not TEST_DATASET_FULL in MetadataCatalog:
        register_coco_instances(
            TEST_DATASET_FULL,
            {},
            PATH_TEST_FULL_JSON,
            PATH_TEST_FULL_IMAGES,
        )
        
    # single point dataset
    if not SINGLE_POINT_DATASET in MetadataCatalog:
        DatasetCatalog.register(SINGLE_POINT_DATASET, register_function_singe_point_dataset)
        MetadataCatalog.get(SINGLE_POINT_DATASET).set(thing_classes = [CELL])


    




        