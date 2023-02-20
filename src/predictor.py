import torch, detectron2
import os
import numpy as np
from skimage import io

from detectron2.engine import DefaultPredictor

from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.evaluation import DatasetEvaluator

from utils.config_builder import get_config
from utils.register_datasets import register_datasets
from src.globals import *

def create_dir_to_path(path, dir_name):
    if not os.path.exists(os.path.join(path, dir_name)):
        os.mkdir(os.path.join(path, dir_name))

def predict_image(image, predictor):

    outputs = predictor(image)
    masks = outputs["instances"].pred_masks
    mask_shape = masks[0].shape
    mask = torch.zeros(mask_shape).to("cuda")

    for id in range(len(masks)):
        mask = torch.max((id + 1) * masks[id],mask)

    return mask.detach().cpu().numpy()


def get_acdc_filenames():
    
    filenames = []
    
    return filenames

def load_acdc_images(root, filename):        
        images = io.imread(os.path.join(root, filename))
        images = [np.stack([image,image,image]).transpose(1, 2, 0) for image in images]
        return images


def predict_acdc_position(cfg, acdc_path, image_suffix):

    create_dir_to_path(PROJECT_ROOT, "predictions")
    create_dir_to_path(os.path.join(PROJECT_ROOT, "predictions"), acdc_path)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    for root, dirs, files in os.walk(os.path.join(DATA_PATH, "raw_data", acdc_path)):
        for filename in files:
            if filename.find(image_suffix) != -1:
                images = load_acdc_images(root, filename)
                masks = []
                print(images[0].shape)
                for image in images:
                    mask = predict_image(image, predictor)
                    masks.append(mask)

                predictions = np.stack(masks).astype(int)

                with open(
                    os.path.join(PROJECT_ROOT,"predictions", acdc_path, filename.replace(image_suffix, "") + "_segm.npz")
                        ,
                        "wb",
                    ) as file:
                    np.savez_compressed(file, predictions)


if __name__ == "__main__":
    register_datasets()
    config_name = "chlamy_full"
    cfg = get_config(config_name)
    predict_acdc_position(cfg=cfg, acdc_path="chlamy", image_suffix="_T_PMT_T1.tif")
