import torch, detectron2

# import some common libraries
import numpy as np
import os, json, cv2, random

from detectron2.engine import DefaultTrainer


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()