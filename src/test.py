import torch, detectron2


# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
#from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data import MetadataCatalog, DatasetCatalog
#from detectron2.data.datasets import register_coco_instances
#from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def test(cfg, dataset):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(dataset, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, dataset)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))