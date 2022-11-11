import sys
sys.path.append("..")

import wandb
import math

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.modeling import build_model

from globals import *
from test import do_test
from train import do_train
from active_learning.active_learning_dataset import ActiveLearingDataset
from active_learning.query_strategies import *




class ActiveLearningTrainer:
    
    def __init__(self, cfg, is_test_mode=False):
        self.cfg = cfg

        # initialize weights and biases
        if is_test_mode:
            wandb.init(project="activeCell-ACDC", sync_tensorboard=True, mode="disabled")
        else:
            wandb.init(project="activeCell-ACDC", sync_tensorboard=True)
        
        self.logger = setup_logger(output="./log/main.log")
        self.logger.setLevel(10)
        
        self.al_dataset = ActiveLearingDataset(cfg)   
        self.model = build_model(cfg)
        self.query_strategy = GTknownSampler(cfg)
        
        
    def __del__(self):
        wandb.run.finish()
    
    def step(self, resume):
        
        len_ds_train = len(DatasetCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        print("lenght of train data set: {}".format(len_ds_train))
        self.cfg.SOLVER.MAX_ITER = min(400 + len_ds_train*5, 1000)
        self.cfg.SOLVER.STEPS = [math.ceil(self.cfg.SOLVER.MAX_ITER/3),math.ceil(2*self.cfg.SOLVER.MAX_ITER/3)]
        
        if not resume:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            
        do_train(self.cfg, self.model, self.logger,resume=resume)
        result = do_test(self.cfg, self.model, self.logger)
        wandb.log(
            {
                "active_step_bbox_ap": result['bbox']['AP'],
                "active_step_segm_ap": result['segm']['AP']
            })
        

        sample_ids = self.query_strategy.sample(self.model, self.al_dataset.unlabeled_ids)
        self.al_dataset.update_labeled_data(sample_ids)
        
    
    def run(self):
        try:
            for i in range(self.cfg.AL.MAX_LOOPS):
                self.step(resume=False)    #(i>0))
        except Exception as e:
            wandb.run.finish()
            raise e