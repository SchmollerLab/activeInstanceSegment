import os, sys

import random as rd
import numpy as np
import torch
import cv2

import shutil

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import detectron2.data.transforms as T

from src.globals import *
from src.active_learning.query_strategies.query_strategy import QueryStrategy
from utils.register_datasets import register_by_ids

class RandomSampler(QueryStrategy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.strategy = "random"
        self.clean_output_dir()

    def sample(self, cfg, ids):
        """Sample datapoints using randomly

        Parameters
        ----------
        cfg
            Detectron2 config file
        ids
            id pool from which ids are sampled
        """
        num_samples = self.cfg.AL.INCREMENT_SIZE
        rd.seed(cfg.SEED)
        if len(ids) > num_samples:
            samples = rd.sample(ids, num_samples)
        else:
            samples = ids

        with open(
            os.path.join(
                self.cfg.AL.OUTPUT_DIR,
                self.strategy,
                f"{self.strategy}_samples{str(self.counter)}.txt",
            ),
            "w",
        ) as file:
            file.write("\n".join(samples))
        self.counter += 1
        return samples