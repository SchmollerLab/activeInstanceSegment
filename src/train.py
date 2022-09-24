import torch, detectron2

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
#from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data import MetadataCatalog, DatasetCatalog
#from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.data import build_detection_test_loader

def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()