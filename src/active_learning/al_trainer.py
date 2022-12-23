import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

import wandb
import math
import yaml

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.modeling import build_model

from src.globals import *
from src.register_datasets import get_dataset_name
from src.test import do_test
from src.train import do_train
from src.active_learning.al_dataset import ActiveLearingDataset
from src.active_learning.query_strategies import *
from src.active_learning.mc_dropout_sampler import MCDropoutSampler


class ActiveLearningTrainer:
    def __init__(self, cfg, is_test_mode=False):
        self.cfg = cfg

        self.is_test_mode = is_test_mode

        self.logger = setup_logger(output="./log/main.log")
        self.logger.setLevel(10)

    def __del__(self):
        wandb.run.finish()

    def step(self, resume):

        if not resume:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        else:
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "best_model.pth")

        self.cfg = do_train(self.cfg, self.logger, resume=resume)
        result = do_test(self.cfg, logger=self.logger)
        wandb.log(
            {
                "al": {
                    "bbox_ap": result["bbox"]["AP"],
                    "segm_ap": result["segm"]["AP"],
                    "used_data_points": self.al_dataset.get_len_labeled(),
                }
            }
        )

        print("test active learning", (result["segm"]["AP"] + result["bbox"]["AP"]) / 2)
        sample_ids = self.query_strategy.sample(self.cfg, self.al_dataset.unlabeled_ids)
        self.al_dataset.update_labeled_data(sample_ids)

    def run(self, dataset, query_strat):

        # initialize weights and biases
        if self.is_test_mode:
            wandb.init(
                project="activeCell-ACDC",
                name=dataset + "_" + query_strat,
                sync_tensorboard=True,
                mode="disabled",
            )
        else:
            wandb.init(
                project="activeCell-ACDC",
                name=dataset + "_" + query_strat,
                sync_tensorboard=True,
            )

        # define strategy
        if query_strat == RANDOM:
            self.query_strategy = RandomSampler(self.cfg)
        elif query_strat == KNOWN_VALIDATION:
            self.query_strategy = GTknownSampler(self.cfg)
        elif query_strat == MC_DROPOUT:
            self.query_strategy = MCDropoutSampler(self.cfg)
        else:
            raise Exception("Query strategy {} not defined".format(query_strat))

        # define al dataset and specify what dataset to use
        self.cfg.AL.DATASETS.TRAIN_UNLABELED = get_dataset_name(dataset, DATASETS_DSPLITS[dataset][0])
        #self.cfg.DATASETS.TRAIN = (get_dataset_name(dataset, DATASETS_DSPLITS[dataset][0],))
        #self.cfg.DATASETS.TEST = (get_dataset_name(dataset, TEST),)
        self.al_dataset = ActiveLearingDataset(self.cfg)
        wandb.config.update(yaml.load(self.cfg.dump()))
        try:
            for i in range(self.cfg.AL.MAX_LOOPS):
                self.step(resume=(i > 0))
        except Exception as e:
            wandb.run.finish()
            raise e
        wandb.run.finish()
