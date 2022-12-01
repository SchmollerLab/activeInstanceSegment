
########################################################################
#                                DATA                                  #
########################################################################

### PATHS
import os

BASE_DATA_PATH =  "./data/"
    
REL_PATH_TRAIN_JSON = "train/cell_acdc_coco_ds.json"
REL_PATH_TRAIN_IMAGES = "train/images"
REL_PATH_TEST_JSON = "test/cell_acdc_coco_ds.json"
REL_PATH_TEST_IMAGES = "test/images"
PATH_PIPELINE_CONFIGS = "./pipeline_configs"

ACDC_SMALL = "acdc_small"
ACDC_LARGE = "acdc_large"
CELLPOSE = "cellpose"

LIST_DATASETS = [ACDC_LARGE, ACDC_SMALL, CELLPOSE]

### NAMES REGISTERED DATASETS
TRAIN_DATASET_FULL = "cell_acdc_train"
TEST_DATASET_FULL = "cell_acdc_test"
SINGLE_POINT_DATASET = "cell_acdc_train_single_sample"
VALIDATION_DATASET_SLIM = "cell_acdc_validation_slim"

### class label
CELL = 'cell'

TRAIN = "train"
TEST = "test"
### AL Methods
RANDOM = "random"
KNOWN_VALIDATION = "known_validation"
