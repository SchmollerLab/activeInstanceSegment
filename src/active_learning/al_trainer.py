import sys
import os

import wandb
import math
import yaml
import copy


from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.modeling import build_model

from src.globals import *
from utils.register_datasets import get_dataset_name
from src.test import do_test
from src.train import do_train
from src.active_learning.al_dataset import ActiveLearingDataset
from src.active_learning.query_strategies import *
from src.active_learning.mc_dropout_sampler import MCDropoutSampler
from src.active_learning.hybrid_sampler import HybridSampler
from src.active_learning.tta_sampler import TTASampler


class ActiveLearningTrainer:
    def __init__(self, cfg, cur_date="", debug_mode=False):
        self.cfg = cfg
        self.debug_mode = debug_mode

        self.logger = setup_logger(output="./log/main.log")
        self.logger.setLevel(10)

        try:
            os.makedirs(self.cfg.AL.OUTPUT_DIR)
        except:
            print(f"{self.cfg.AL.OUTPUT_DIR} already exists")

        query_strat = cfg.AL.QUERY_STRATEGY

        # define strategy
        if query_strat == RANDOM:
            self.query_strategy = RandomSampler(self.cfg)
        elif query_strat == MC_DROPOUT:
            self.query_strategy = MCDropoutSampler(self.cfg)
        elif query_strat == HYBRID:
            self.query_strategy = HybridSampler(self.cfg)
        elif query_strat == TTA:
            self.query_strategy = TTASampler(self.cfg)
        else:
            raise Exception("Query strategy {} not defined".format(query_strat))

        self.al_dataset = ActiveLearingDataset(self.cfg)

        # initialize weights and biases
        print("initializing wandb")
        wandb.init(
            project="activeCell-ACDC",
            name=str(query_strat + "_" + cur_date + "_" + os.uname()[1]).split("-")[0],
            sync_tensorboard=False,
            mode="disabled" if self.debug_mode else None,
        )

        wandb.config.update(
            yaml.load(self.cfg.dump(), Loader=yaml.Loader), allow_val_change=True
        )

        # hyperparameters for model training
        self.epochs = self.cfg.AL.MAX_TRAINING_EPOCHS

        len_ds = self.al_dataset.get_len_labeled()
        steps_per_epoch = int(len_ds / self.cfg.SOLVER.IMS_PER_BATCH)

        self.cfg.TEST.EVAL_PERIOD = max(
            6 * self.cfg.SOLVER.IMS_PER_BATCH * steps_per_epoch, 20
        )

        self.model_name = f"{self.query_strategy.strategy}/last_model{self.al_dataset.get_len_labeled()}.pth"

    def __del__(self):
        wandb.finish()

    def step(self, resume):

        result = do_train(
            self.cfg,
            self.logger,
            resume=resume,
            custom_max_iter=self.steps_per_epoch * self.epochs,
        )

        model_path = os.path.join(self.cfg.OUTPUT_DIR, "last_model.pth")
        os.system(
            f"cp {model_path} {os.path.join(self.cfg.AL.OUTPUT_DIR, self.odel_name)}"
        )
        wandb.log(
            {
                "al": {
                    "bbox_ap": result["bbox"]["AP"],
                    "segm_ap": result["segm"]["AP"],
                    "used_data_points": self.al_dataset.get_len_labeled(),
                    "num_labeled_objects": self.al_dataset.get_num_objects(),
                }
            }
        )
        with open(
            os.path.join(
                self.cfg.AL.OUTPUT_DIR,
                self.query_strategy.strategy,
                f"{self.query_strategy.strategy}_traindata{str(self.al_dataset.get_len_labeled())}.txt",
            ),
            "w",
        ) as file:
            file.write("\n".join(self.al_dataset.labeled_ids))

        sample_ids = self.query_strategy.sample(self.cfg, self.al_dataset.unlabeled_ids)
        self.al_dataset.update_labeled_data(sample_ids)
        return len(sample_ids) > 0

    def run(self):
        try:
            for i in range(self.cfg.AL.MAX_LOOPS):
                if not self.step(resume=not self.cfg.AL.RETRAIN):
                    break
        except Exception as e:
            wandb.finish()
            raise e
