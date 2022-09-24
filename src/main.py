import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger(output="./main.log")

# import some common libraries
import numpy as np
import os, json, cv2, random



# import some common detectron2 utilities
from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
#from detectron2.engine import DefaultTrainer
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.data import build_detection_test_loader

from train import train
from test import test


def make(model, train_dataset, test_dataset, num_dataloader_worker, batch_size, base_lr, max_iter ):
    
    #empty gpu cache
    torch.cuda.empty_cache()
    
    
    cfg = get_cfg()
    cfg.merge_from_file( model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = (train_dataset,) # TODO check if test ds works here
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = num_dataloader_worker
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batch_size  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # TODO: what is this hyperparam?
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class

    return cfg


TRAIN_DATASET = "cell_acdc_train"
TEST_DATASET = "cell_acdc_test"

MODEL ="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # TODO customize
BATCH_SIZE = 2
NUM_DL_WORKER = 2
BASE_LR = 0.00025
MAX_ITER = 1000


if __name__ == "__main__":

    # register new data
    register_coco_instances(TRAIN_DATASET, {}, "./segmentationMicroscopy/data/dataInCOCO/train/cell_acdc_coco_ds.json", "./segmentationMicroscopy/data/dataInCOCO/train/images")
    register_coco_instances(TEST_DATASET, {}, "./segmentationMicroscopy/data/dataInCOCO/test/cell_acdc_coco_ds.json", "./segmentationMicroscopy/data/dataInCOCO/test/images")



    cfg = make(
        model=MODEL,
        train_dataset=TRAIN_DATASET,
        test_dataset=TEST_DATASET,
        num_dataloader_worker=NUM_DL_WORKER,
        batch_size=BATCH_SIZE,
        base_lr=BASE_LR,
        max_iter=MAX_ITER
    )

    train(cfg)
    test(cfg,TEST_DATASET)