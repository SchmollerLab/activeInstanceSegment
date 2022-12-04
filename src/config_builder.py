from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode as CN

import sys
try:
    sys.path.append("./src")
except:
    pass

from globals import *
from register_datasets import get_dataset_name


from active_learning.nn_modules.dropout import *

def build_config(config_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.NAME = config_name
    cfg.AL = CN()
    cfg.AL.DATASETS = CN()
    cfg.AL.DATASETS.TRAIN_UNLABELED = ""
    cfg.AL.MAX_LOOPS = 7
    cfg.AL.INIT_SIZE = 50
    cfg.AL.INCREMENT_SIZE = 50
    cfg.AL.QUERY_STRATEGY = RANDOM
    
    cfg.DATASETS.TRAIN = (get_dataset_name(ACDC_LARGE,TRAIN),)    
    cfg.DATASETS.TEST = (get_dataset_name(ACDC_LARGE,TEST),)
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.0003  # pick a good LR
    cfg.SOLVER.MAX_ITER = 40000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.EARLY_STOPPING_ROUNDS = 5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = "./output/" + cfg.NAME
    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.ROI_HEADS.SOFTMAXES = False
    cfg.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = 0.5
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = 0.5

    
    with open(PATH_PIPELINE_CONFIGS + "/" + cfg.NAME + ".yaml","w") as file:
        file.write(cfg.dump())

def get_config(config_name, path_configs=PATH_PIPELINE_CONFIGS, complete_path=None):
    
    cfg = get_cfg()
    cfg.NAME = " "
    cfg.AL = CN()
    cfg.AL.DATASETS = CN()
    cfg.AL.DATASETS.TRAIN_UNLABELED = ""
    cfg.AL.MAX_LOOPS = 0
    cfg.AL.INIT_SIZE = 0
    cfg.AL.INCREMENT_SIZE = 0
    cfg.AL.QUERY_STRATEGY = ""
    cfg.EARLY_STOPPING_ROUNDS = 0
    
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False
    cfg.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = 0.5
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = 0.5

    if not complete_path:
        file_path = path_configs + "/" + config_name + ".yaml"
    else:
        file_path = complete_path
    cfg.merge_from_file(file_path)
    return cfg

if __name__ == "__main__":

    build_config("acdc_large_full_ds")
    
