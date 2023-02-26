import random as rd
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from src.globals import *
from utils.register_datasets import register_by_ids, get_dataset_name
from src.active_learning.query_strategies import UncertaintySampler

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
from tqdm import tqdm
import operator
import wandb
import json
import time


import torch
from itertools import chain
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import detectron2.data.transforms as T
from baal.bayesian.dropout import patch_module


class TTASampler(UncertaintySampler):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.strategy = "tta"
        self.clean_output_dir()

    def log_results(self,uncertainty_dict, samples):
        with open(os.path.join(self.cfg.AL.OUTPUT_DIR, self.strategy, f"uncertainties{str(self.counter)}.json"),"w") as file:
            json.dump(uncertainty_dict, file)
        
        print("finished with tta sampling.")
        print(
            "min uncertainty: ",
            min(list(uncertainty_dict.values())),
            "\t mean uncertainty: ",
            sum(list(uncertainty_dict.values())) / len(list(uncertainty_dict.values())),
            "\t max uncertainty: ",
            max(list(uncertainty_dict.values())),
        )
        print("worst examples:", samples)
        wandb.log(
            {
                "al":{
                    "min_uncertainty":min(list(uncertainty_dict.values())),
                    "mean_uncertainty":sum(list(uncertainty_dict.values())) / len(list(uncertainty_dict.values())),
                    "max_uncertainty":max(list(uncertainty_dict.values())),
                }
            }
        )

        with open(os.path.join(self.cfg.AL.OUTPUT_DIR, self.strategy, f"{self.strategy}_samples{str(self.counter)}.txt"),"w") as file:
            file.write("\n".join(samples))

    
    def sample(self, cfg, ids, custom_model):

        num_samples = self.cfg.AL.INCREMENT_SIZE
        id_pool = self.presample_id_pool(cfg, ids)
        register_by_ids(
            "ALSampler_DS",
            id_pool,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )

        if not custom_model:
            print(f"loading default model")
            model = build_model(cfg)
            model = patch_module(model)
            model.eval()

            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))
        else:
            print(f"loading custom model")
            model = custom_model

        

        ds_catalog = DatasetCatalog.get("ALSampler_DS")
        uncertainty_dict = {}
        print("running tta sampling...")
        for i in tqdm(range(len(ds_catalog))):

            im_json = ds_catalog[i]
            im = self.load_image(im_json)

            instance_list = self.get_samples(model, im, cfg.AL.NUM_MC_SAMPLES)
            combinded_instances = self.get_combinded_instances(instance_list, iou_thres=0.1)


            height, width = im.shape[:2]
            uncertainty = self.get_uncertainty(combinded_instances, cfg.AL.NUM_MC_SAMPLES, height, width, mode=cfg.AL.OBJECT_TO_IMG_AGG, bbox=False)

            uncertainty_dict[im_json["image_id"]] = float(uncertainty)

        worst_ims = np.argsort(list(uncertainty_dict.values()))[-num_samples:]
        samples = [list(uncertainty_dict.keys())[id] for id in worst_ims]

        self.log_results(uncertainty_dict, samples)

        self.counter += 1
        return samples
    
    def preprocess_image_rotate(self, input_image, model, angle):

        height, width = input_image.shape[:2]
        image = self.aug.get_transform(input_image).apply_image(input_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        rot_image = TF.rotate(image, angle)
        inputs = [{"image": rot_image, "height": height, "width": width}]
        images = model.preprocess_image(inputs)

        return images, inputs


    def get_samples(self, model, input_image, iterrations):

        with torch.no_grad():
            prediction_list = []

            for angle in range(0, 360, int(360/iterrations)):

                images, inputs = self.preprocess_image_rotate(input_image, model, angle)

                features = model.backbone(images.tensor)
                proposals, box_features_pooler = self.get_backbone_roi_proposals(model, images, features)
               
                instances = self.get_instance_detections(model, inputs, images, features, proposals, box_features_pooler)
                instances = self.transform_back(instances, angle)
                prediction_list.append(instances)

            return list(chain.from_iterable(prediction_list))

    def transform_back(self, instances, angle):
        for i in range(len(instances)):
            instances[i]["instances"].pred_masks = TF.rotate(instances[i]["instances"].pred_masks, -angle)

        return instances

if __name__ == "__main__":

    import cProfile
    from utils.config_builder import get_config
    from src.active_learning.al_dataset import ActiveLearingDataset

    wandb.init(
        project="activeCell-ACDC",
        name="",
        sync_tensorboard=True,
        mode="disabled",
    )

    config_name = "acdc_large_al"
    cfg = get_config(config_name)

    cfg.AL.SAMPLE_EVERY = 240
    al_dataset = ActiveLearingDataset(cfg)
    query_strategy = TTASampler(cfg)

    cProfile.run('query_strategy.sample(cfg, al_dataset.unlabeled_ids)')
