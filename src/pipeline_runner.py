import torch, detectron2
from datetime import datetime


# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

logger = setup_logger(output="./log/main.log")

# import some common libraries
import numpy as np
import os, json, cv2, random
import wandb

# import some common detectron2 utilities
from detectron2.modeling import build_model

from argparse import ArgumentParser

try:
    from register_datasets import register_datasets
    from train import do_train
    from test import do_test
    from config_builder import get_config
except:
    from src.register_datasets import register_datasets
    from src.train import do_train
    from src.test import do_test


def run_pipeline(cfg=None):

    model = build_model(cfg)
    # logger.info("Model:\n{}".format(model))

    # initialize weights and biases
    wandb.init(
        project="activeCell-ACDC",
        name=cfg.NAME + "-" + datetime.now().strftime("%d-%b-%Y"),
        sync_tensorboard=True,
    )

    # empty gpu cache
    torch.cuda.empty_cache()
    # run training
    do_train(cfg, model, logger)
    # run testing
    result = do_test(cfg, model, logger)

    model = None
    cfg = None

    wandb.run.finish()

    return result


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="Path to pipeline configuration",
        metavar="FILE",
    )

    args = parser.parse_args()
    filename = args.filename

    register_datasets()
    config_name = filename.split("/")[-1].replace(".yaml", "")
    cfg = get_config(config_name)

    run_pipeline(cfg)
