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

from utils.register_datasets import *
from src.train import do_train
from src.test import do_test
from utils.config_builder import get_config

from src.pipeline_runner import run_pipeline


if __name__ == "__main__":
    config_name = "final_random_al"
    cfg = get_config(config_name)

    register_datasets()

    dataset = cfg.DATASETS.TRAIN[0]

    train_data = DatasetCatalog.get(dataset)

    rd.seed(1337)
    sub_samples = rd.sample(train_data, 200)

    ids = [image_json["image_id"] for image_json in sub_samples]

    register_by_ids(
        dataset + "_subsample",
        ids,
        BASE_DATA_PATH + dataset + "/" + "test" + "/" + REL_PATH_JSON,
        dataset,
    )

    cfg.DATASETS.TRAIN = (dataset + "_subsample",)

    with open(
        "/home/florian/GitRepos/activeCell-ACDC/output/final_random_al/al_output/random/small_train_id.txt",
        "w",
    ) as file:
        file.write("\n".join(ids))
    run_pipeline(config_name, cfg)
