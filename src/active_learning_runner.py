
import json
import random as rd
import matplotlib.image as mpimg
import cv2
import wandb
import torch
import numpy as np

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode as CN
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from globals import *
from visualization.show_image import show_image
from register_datasets import register_datasets, register_by_ids
from test import do_test
from train import do_train
from predict import predict_image_in_acdc


class ActiveLearingDataset:
    
    def __init__(self, cfg):
        """
        
        
        """
        
        self.cfg = cfg
        
        register_datasets()

        # get ids of all images
        self.unlabeled_ids = [image["image_id"] for image in DatasetCatalog.get(cfg.AL.DATASETS.TRAIN_UNLABELED)]
        self.labeled_ids = []
        
        self.unlabeled_data_name = "temp_unlabeled_data_al"
        self.labeled_data_name = "temp_labeled_data_al"
        
        self.init_size = cfg.AL.INIT_SIZE
        self.increment_size = cfg.AL.INCREMENT_SIZE
        
        # set seed
        rd.seed(1337)
        sample_ids = rd.sample(self.unlabeled_ids, self.init_size)
        self.update_labeled_data(sample_ids)
        self.get_labeled_dataset()
        self.get_unlabled_dataset()
        
    
    def remove_data_from_catalog(self,name):
        
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        
        
    def get_labeled_dataset(self):
        self.remove_data_from_catalog(self.labeled_data_name)
        register_by_ids(self.labeled_data_name, self.labeled_ids)
        self.cfg.DATASETS.TRAIN = (self.labeled_data_name,)
    
    def get_unlabled_dataset(self):
        self.remove_data_from_catalog(self.unlabeled_data_name)
        register_by_ids(self.unlabeled_data_name,self.unlabeled_ids)
        self.cfg.AL.DATASETS.TRAIN_UNLABELED = self.unlabeled_data_name
    
    def update_labeled_data(self, sample_ids):
        print("update_labeled_data")
        # check if sample_ids are in unlabeled_ids
        if not (set(sample_ids) <= set(self.unlabeled_ids)):
            raise Exception("Some ids ({}) in sample_ids are not contained in unlabeled data pool: {}".format(len(list(set(sample_ids) - set(self.unlabeled_ids))),list(set(sample_ids) - set(self.unlabeled_ids))[:5])) 
        
        self.labeled_ids += sample_ids
        print(self.labeled_ids)
        self.unlabeled_ids = list(set(self.unlabeled_ids) - set(sample_ids))
        print(self.unlabeled_ids)
        
        self.get_labeled_dataset()
        self.get_unlabled_dataset()



class QueryStrategy(object):
    
    def __init__(self,cfg):
        
        self.cfg = cfg
        
    
    def sample(self,model, ids):
        pass
    
class RandomSampler(QueryStrategy):
    
    def sample(self,model, ids):
        num_samples = self.cfg.AL.INCREMENT_SIZE        
        samples = rd.sample(ids, num_samples)
        return samples

class GTknownSampler(QueryStrategy):
    
    def sample(self, model, ids):
        num_samples = self.cfg.AL.INCREMENT_SIZE        
        id_pool = rd.sample(ids, 30)
        
        register_by_ids("GTknownSampler_DS",id_pool)

        
        evaluator = COCOEvaluator("GTknownSampler_DS", output_dir=self.cfg.OUTPUT_DIR)
        data_loader = build_detection_test_loader(self.cfg, "GTknownSampler_DS")
        inference_on_dataset(model, data_loader, evaluator)


        result_array = []
        for image_id in [image["image_id"] for image in DatasetCatalog.get("GTknownSampler_DS")]:
            result = evaluator.evaluate(image_id)
            result_array.append(result)

        aps = np.array([result['segm']['AP'] for result in result_array])
        samples = np.argsort(aps)[:num_samples]

        return samples
    

class ActiveLearningTrainer:
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        # initialize weights and biases
        wandb.init(project="activeCell-ACDC", sync_tensorboard=True)
        
        self.logger = setup_logger(output="./log/main.log")
        self.logger.setLevel(0)
        
        self.al_dataset = ActiveLearingDataset(cfg)   
        self.model = build_model(cfg)
        self.query_strategy = GTknownSampler(cfg)
        
        
    def __del__(self):
        wandb.run.finish()
    
    def step(self, resume):
        
        len_ds_train = len(DatasetCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        print("lenght of train data set: {}".format(len_ds_train))
        self.cfg.SOLVER.MAX_ITER = len_ds_train*1#20
        self.cfg.SOLVER.STEPS = []#[len_ds_train*10]
        
        do_train(self.cfg, self.model, self.logger,resume=resume)
        #result = do_test(self.cfg, self.model, self.logger)

        self.al_dataset.update_labeled_data(self.query_strategy.sample(self.model, self.al_dataset.unlabeled_ids))
        
    
    def run(self):
        try:
            for i in range(self.cfg.AL.MAX_LOOPS):
                self.step(resume=(i>0))
        except Exception as e:
            wandb.run.finish()
            raise e
            

def get_config(config_name):
    
    cfg = get_cfg()
    cfg.NAME = " "
    cfg.AL = CN()
    cfg.AL.DATASETS = CN()
    cfg.AL.DATASETS.TRAIN_UNLABELED = ""
    cfg.AL.MAX_LOOPS = 0
    cfg.AL.INIT_SIZE = 0
    cfg.AL.INCREMENT_SIZE = 0
    cfg.AL.QUERY_STRATEGY = ""
    cfg.WARMUP_ITERS = 0
    
    file_path = "src/pipeline_configs/" + config_name + ".yaml"
    cfg.merge_from_file(file_path)
    return cfg

if __name__ == "__main__":

    cfg = get_config("al_pipeline_config")
    al_trainer = ActiveLearningTrainer(cfg)
    al_trainer.run()