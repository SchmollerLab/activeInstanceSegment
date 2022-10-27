import torch, detectron2

#TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
#CUDA_VERSION = torch.__version__.split("+")[-1]
#print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
#print("detectron2:", detectron2.__version__)


# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

logger = setup_logger(output="./log/main.log")
logger.setLevel(0)
# import some common libraries
import numpy as np
import os, json, cv2, random
import wandb

# import some common detectron2 utilities
from detectron2.modeling import build_model
from detectron2.config import get_cfg

from argparse import ArgumentParser

try:
    from register_datasets import register_datasets
    from train import do_train
    from test import do_test
except:
    from src.register_datasets import register_datasets
    from src.train import do_train
    from src.test import do_test


def get_config(filename):
    
    cfg = get_cfg()
    cfg.merge_from_file(filename)
    return cfg
 

    
def run_pipeline(cfg=None):
    
    model = build_model(cfg)
    #logger.info("Model:\n{}".format(model))
    
    # initialize weights and biases
    wandb.init(project="activeCell-ACDC", sync_tensorboard=True)
    
    # empty gpu cache
    torch.cuda.empty_cache()
    # run training
    do_train(cfg, model, logger)
    #run testing
    result = do_test(cfg, model, logger)

    model = None
    cfg = None

    wandb.run.finish()
    
    return result
        
        
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="write report to FILE", metavar="FILE")

    args = parser.parse_args()
    filename = args.filename
    
    
    register_datasets()
    cfg = get_config(filename)
    
    cfg.OUTPUT_DIR = "./output/" + filename.replace(".yaml","").replace("src/pipeline_configs/","")
    
    run_pipeline(cfg)
    