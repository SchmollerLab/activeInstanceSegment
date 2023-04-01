import json
import random as rd
import matplotlib.image as mpimg
import cv2
import wandb
import torch
import os
import logging
import pandas as pd
import collections
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer

from src.globals import *
from utils.visualization.show_image import show_image
from utils.register_datasets import register_datasets, get_dataset_name
from utils.config_builder import get_config
from utils.notebook_utils import *


from src.test import do_test
from src.active_learning.al_trainer import *
from src.active_learning.mc_dropout_sampler import *
from src.active_learning.tta_sampler import *


logger = setup_logger(output="./log/main.log",name="null_logger") 
logger.addHandler(logging.NullHandler())
logging.getLogger('detectron2').setLevel(logging.WARNING)
logging.getLogger('detectron2').addHandler(logging.NullHandler())


def val2dicts(val):

    preds = []
    for v in val:
        pred_masks = v["pred_masks"].detach().cpu().numpy()
        pred_classes = v["pred_classes"].detach().cpu().numpy()

        preds.append({
            "pred_masks": pred_masks,
            "pred_classes": pred_classes,
        })

    return preds

def get_uncertainties(im_json, model, query_strategy):

    im = query_strategy.load_image(im_json)
    instance_list = query_strategy.get_samples(model, im, cfg.AL.NUM_MC_SAMPLES)
    combinded_instances = query_strategy.get_combinded_instances(instance_list)


    height, width = im.shape[:2]
    agg_uncertainty = query_strategy.get_uncertainty(combinded_instances, cfg.AL.NUM_MC_SAMPLES, height, width, mode=cfg.AL.OBJECT_TO_IMG_AGG)

    
    uncertainties = []

    for key, val in combinded_instances.items():

        val_len = torch.tensor(len(val)).to("cuda")

        if query_strategy.cfg.MODEL.ROI_HEADS.NUM_CLASSES > 1:
            u_sem = query_strategy.get_semantic_certainty(val, device = "cuda").detach().cpu().numpy()
        else:
            u_sem = 0
        u_mask = query_strategy.get_mask_certainty(val, height, width, val_len, device="cuda").detach().cpu().numpy()
        u_box = query_strategy.get_box_certainty(val, val_len, device="cuda").detach().cpu().numpy()
        u_det = query_strategy.get_detection_certainty(cfg.AL.NUM_MC_SAMPLES, val_len, device="cuda").detach().cpu().numpy()
        
        

        cpu_val = val2dicts(val)

        uncertainties.append({
            "val": cpu_val,
            "u_sem": u_sem,
            "u_mask": u_mask,
            "u_box": u_box,
            "u_det": u_det,

        })


    
    return uncertainties, agg_uncertainty

results_path = "jupyter_notebooks/results"

dataset = ACDC_LARGE_CLS
config_name = "classes_acdc_large_al"

model_path = "/mnt/activeCell-ACDC/al_output/classes_acdc_large_al/random"

register_datasets()

test_data = DatasetCatalog.get("acdc_large_cls_test_slim")


wandb.init(
    project="activeCell-ACDC",
    name="",
    sync_tensorboard=True,
    mode="disabled",
)


default_cfg = get_config(config_name)
cfg = default_cfg

cfg.OUTPUT_DIR = "./al_output/classes_acdc_large_al"
cfg.AL.OBJECT_TO_IMG_AGG = "mean"

mc_strategy = MCDropoutSampler(cfg)
tta_strategy = TTASampler(cfg)




records_models = []

num_mc_samples = 20
print(num_mc_samples)
cfg_test = cfg
cfg_test.AL.NUM_MC_SAMPLES = num_mc_samples
mc_strategy = MCDropoutSampler(cfg_test)

num_train_data = 270


for dropout_prob in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    cfg_test = cfg
    
    cfg_test.MODEL.ROI_HEADS.DROPOUT_PROBABILITY = dropout_prob
    cfg_test.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = dropout_prob
    cfg_test.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = dropout_prob
    
    mc_strategy = MCDropoutSampler(cfg_test)
    mc_model = load_model(cfg_test, os.path.join(model_path, f"best_model{str(num_train_data)}.pth"))
    mc_model = patch_module(mc_model)
    for im_json in tqdm(test_data):
        single_im_unc = []
        for run_id in range(10):
            uncertainties, agg_uncertainty = get_uncertainties(im_json, mc_model, mc_strategy)
            records_models.append({
                "num_mc_samples": num_mc_samples,
                "dropout_prob": dropout_prob,
                "model_train_size": num_train_data,
                "image_id": im_json["image_id"],
                "run_id": run_id,
                "agg_uncertainty": agg_uncertainty,
                #"uncertainties": uncertainties,
            })
            
        
    
    
            
df = pd.DataFrame.from_records(records_models)
df.to_csv(os.path.join(results_path,"uncertainties_vs_drop_prob.csv"))