import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

logger = setup_logger(output="./main.log")

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model

from train import do_train
from test import do_test


def make(
    model,
    train_dataset,
    test_dataset,
    num_dataloader_worker,
    batch_size,
    base_lr,
    max_iter,
):

    # empty gpu cache
    torch.cuda.empty_cache()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.DATALOADER.NUM_WORKERS = num_dataloader_worker
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # decay lr
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512) TODO: what is this hyperparam?
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class
    cfg.OUTPUT_DIR = "./output"

    model = build_model(cfg)

    return cfg, model


TRAIN_DATASET = "cell_acdc_train"
TEST_DATASET = "cell_acdc_test"

MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"  # TODO customize
BATCH_SIZE = 2
NUM_DL_WORKER = 2
BASE_LR = 0.00025
MAX_ITER = 10


if __name__ == "__main__":

    # register new data
    register_coco_instances(
        TRAIN_DATASET,
        {},
        "./../data/dataInCOCO/train/cell_acdc_coco_ds.json",
        "./../data/dataInCOCO/train/images",
    )
    register_coco_instances(
        TEST_DATASET,
        {},
        "./../data/dataInCOCO/test/cell_acdc_coco_ds.json",
        "./../data/dataInCOCO/test/images",
    )

    cfg, model = make(
        model=MODEL,
        train_dataset=TRAIN_DATASET,
        test_dataset=TEST_DATASET,
        num_dataloader_worker=NUM_DL_WORKER,
        batch_size=BATCH_SIZE,
        base_lr=BASE_LR,
        max_iter=MAX_ITER,
    )

    logger.info("Model:\n{}".format(model))
    do_train(cfg, model, logger)
    do_test(cfg, model, logger)
