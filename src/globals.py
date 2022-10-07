
########################################################################
#                                DATA                                  #
########################################################################

### PATHS
PATH_DATA_IN_COCO = "./data/dataInCOCO/"
PATH_TRAIN_FULL_JSON = PATH_DATA_IN_COCO + "train/cell_acdc_coco_ds.json"
PATH_TRAIN_FULL_IMAGES = PATH_DATA_IN_COCO + "train/images"
PATH_TEST_FULL_JSON = PATH_DATA_IN_COCO + "test/cell_acdc_coco_ds.json"
PATH_TEST_FULL_IMAGES = PATH_DATA_IN_COCO + "test/images"

### NAMES REGISTERED DATASETS
TRAIN_DATASET_FULL = "cell_acdc_train"
TEST_DATASET_FULL = "cell_acdc_test"
SINGLE_POINT_DATASET = "cell_acdc_train_single_sample"

### class label
CELL = 'cell'