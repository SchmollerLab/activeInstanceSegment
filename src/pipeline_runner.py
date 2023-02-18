import torch, detectron2
from datetime import datetime


# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, sys, json, cv2, random
from datetime import date
import wandb
import yaml

# import some common detectron2 utilities
from detectron2.modeling import build_model

from argparse import ArgumentParser

from utils.register_datasets import register_datasets
from src.train import do_train
from src.test import do_test
from utils.config_builder import get_config


def run_pipeline(config_name, cfg=None, cur_date=""):

    logger = setup_logger(output="./log/main.log")
    # logger.info("Model:\n{}".format(model))

    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)

    # initialize weights and biases
    if not running_on_server:
        wandb.init(
            project="activeCell-ACDC",
            name=config_name,
            sync_tensorboard=True,
            mode="disabled",
        )
    else: 
        wandb.init(project="activeCell-ACDC", name=str(cur_date + "_" + config_name + "_" +  os.uname()[1]).split("-")[0], sync_tensorboard=True)

    # empty gpu cache
    torch.cuda.empty_cache()
    # run trainingrunning_on_server
    wandb.config.update(yaml.load(cfg.dump(),Loader=yaml.Loader))
    do_train(cfg, logger=logger)
    # run testing
    result = do_test(cfg, logger=logger)
    wandb.log({"max_ap": (result["segm"]["AP"] + result["bbox"]["AP"]) / 2})
    cfg = None

    wandb.run.finish()

    return result

def grid_search(cfg):


    #print(cfg.MODEL.BACKBONE.FREEZE_AT)
    #print(cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE)

    max_ap = 0
    max_base_lr = 0
    max_lr_scheduler_name = ""
    max_steps = ()

    cur_date = str(date.today().month) + str(date.today().day)

    for base_lr in [0.001,0.0005,0.0003,0.0001]:
        print("cfg.SOLVER.BASE_LR", base_lr)
        cfg.SOLVER.BASE_LR = base_lr
        for lr_scheduler_name, steps in zip(["WarmupCosineLR", "WarmupMultiStepLR", "WarmupMultiStepLR", "WarmupMultiStepLR"],[(),(),(60000),(40000,80000)]):
            
            print("cfg.SOLVER.LR_SCHEDULER_NAME:", lr_scheduler_name)
            cfg.SOLVER.LR_SCHEDULER_NAME = lr_scheduler_name

            print("cfg.SOLVER.STEPS", steps)
            cfg.SOLVER.STEPS = steps
            
            result = run_pipeline(config_name, cfg, cur_date=cur_date)
            ap = (result["segm"]["AP"] + result["bbox"]["AP"]) / 2

            if ap > max_ap:
                max_ap = ap
                max_base_lr = base_lr
                max_lr_scheduler_name = lr_scheduler_name
                max_steps = steps
    
    cfg.SOLVER.LR_SCHEDULER_NAME = max_lr_scheduler_name
    cfg.SOLVER.BASE_LR = max_base_lr
    cfg.SOLVER.STEPS = max_steps

    for anchor_sizes in [[[32],[64],[128],[256],[512],],[[16],[32],[64],[128],[256],]]:
        print("cfg.MODEL.ANCHOR_GENERATOR.SIZES", anchor_sizes)
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = anchor_sizes
        for iou_thresh in [0.5,0.4,0.6]:
            print("cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS", iou_thresh)
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = iou_thresh
            for score_thresh_test in [0.05, 0, 0.1, 0.9]:
                print("cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh_test)
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
                for clip_gradients in [False, True]:
                    print("cfg.SOLVER.CLIP_GRADIENTS.ENABLED", clip_gradients)
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = clip_gradients
                    result = run_pipeline(config_name, cfg, cur_date=cur_date)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config_filename",
        help="Path to pipeline configuration",
        metavar="FILE",
    )

    parser.add_argument(
        "-gs",
        "--gridsearch",
        dest="do_grid_search",
        help="Flag if gridsearch should be done",
        action='store_true',
    )

    args = parser.parse_args()
    config_filename = args.config_filename
    do_grid_search = args.do_grid_search

    register_datasets()
    config_name = config_filename.split("/")[-1].replace(".yaml", "")
    cfg = get_config(config_name, complete_path=config_filename)

    if do_grid_search:
        grid_search(cfg)
    else:
        run_pipeline(config_name, cfg)
