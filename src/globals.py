########################################################################
#                                DATA                                  #
########################################################################

### PATHS
import os

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
DATA_PATH = os.getenv("DATA_PATH")
BASE_DATA_PATH = DATA_PATH + "/"  # "./data/"

REL_PATH_JSON = "cell_acdc_coco_ds.json"
REL_PATH_IMAGES = "images"

PATH_PIPELINE_CONFIGS = "./pipeline_configs"

ACDC_SMALL = "acdc_small"
ACDC_LARGE = "acdc_large"
ACDC_LARGE_CLS = "acdc_large_cls"
CHLAMY_DATASET = "chlamy_dataset"
SHMOO_YEAST = "shmoo_yeast"
SHMOO_YEAST_FULL = "shmoo_yeast_full"

CELLPOSE = "cellpose"
# ACDC_LAST_IMAGES = "acdc_last_images"

# LIST_DATASETS = [ACDC_LARGE, ACDC_SMALL] # [ACDC_LARGE, ACDC_SMALL, CELLPOSE, ACDC_LAST_IMAGES]
DATASETS_DSPLITS = {
    # ACDC_SMALL: ["train", "test"],
    ACDC_LARGE: ["train", "test"],
    ACDC_LARGE_CLS: ["train", "test"],
    CHLAMY_DATASET: ["train", "test"],
    CELLPOSE: ["train", "test"],
    SHMOO_YEAST: ["train", "test"],
    SHMOO_YEAST_FULL: ["train", "test"],
}

### NAMES REGISTERED DATASETS
TRAIN_DATASET_FULL = "cell_acdc_train"
TEST_DATASET_FULL = "cell_acdc_test"
SINGLE_POINT_DATASET = "cell_acdc_train_single_sample"
VALIDATION_DATASET_SLIM = "cell_acdc_validation_slim"

### class label
CELL = "cell"

# TRAIN = "train"
# TEST = "test"
### AL Methods
RANDOM = "random"
KNOWN_VALIDATION = "known_validation"
MC_DROPOUT = "mc_dropout"
HYBRID = "hybrid"
TTA = "tta"
