import os
import sys

sys.path.append("..")

import random as rd
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from globals import *
from register_datasets import register_by_ids, get_dataset_name
from active_learning.query_strategies import QueryStrategy


import numpy as np
import torch
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


class MCDropoutSampler(QueryStrategy):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.strategy = "mc_dropout"
        self.clean_output_dir()

    def sample(self, cfg, ids):

        num_samples = self.cfg.AL.INCREMENT_SIZE
        #num_samples = 2**self.counter

        id_pool = ids  # rd.sample(ids, min(60,len(ids)))

        #rand_int = rd.randint(0,30)
        #id_pool = list(filter(lambda x: (int(x.split("_")[-1]) + rand_int) % 30 == 0, ids))

        register_by_ids(
            "MCDropoutSampler_DS",
            id_pool,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )

        model = build_model(cfg)
        model = patch_module(model)
        model.eval()

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))

        ds_catalog = DatasetCatalog.get("MCDropoutSampler_DS")
        uncertainty_dict = {}
        print("running mc dropout sampling...")
        for i in tqdm(range(len(ds_catalog))):

            im_json = ds_catalog[i]
            im = cv2.imread(im_json["file_name"])
            outputs = self.get_mc_dropout_samples(cfg, model, im, 10)
            predictions = self.get_observations(outputs)
            height, width = im.shape[:2]
            uncertainty = self.get_uncertainty(predictions, 10, height, width)

            uncertainty_dict[im_json["image_id"]] = float(uncertainty)

        with open(os.path.join(self.cfg.AL.OUTPUT_DIR, self.strategy, f"uncertainties{str(self.counter)}.json"),"w") as file:
            json.dump(uncertainty_dict, file)
        worst_ims = np.argsort(list(uncertainty_dict.values()))[:num_samples]
        samples = [list(uncertainty_dict.keys())[id] for id in worst_ims]
        print("finished with mc dropout sampling.")
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
        self.counter += 1
        return samples

    def get_mc_dropout_samples(self, cfg, model, input_image, iterrations):

        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        with torch.no_grad():

            height, width = input_image.shape[:2]
            image = aug.get_transform(input_image).apply_image(input_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]

            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)

            proposals, _ = model.proposal_generator(images, features, None)
            features_ = [features[f] for f in model.roi_heads.box_in_features]

            box_features_pooler = model.roi_heads.box_pooler(
                features_, [x.proposal_boxes for x in proposals]
            )
            prediction_list = []
            for _ in range(iterrations):
                box_features = model.roi_heads.box_head(box_features_pooler)
                predictions = model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = model.roi_heads.box_predictor.inference(
                    predictions, proposals
                )
                pred_instances = model.roi_heads.forward_with_given_boxes(
                    features, pred_instances
                )

                outputs = model._postprocess(pred_instances, inputs, images.image_sizes)
                prediction_list.append(outputs)
            return list(chain.from_iterable(prediction_list))

    def get_observations(self, outputs, iou_thres=0.5):
        """
        To cluster the segmentations for the different Monte-Carlo runs
        """
        observations = {}
        obs_id = 0

        for i in range(len(outputs)):
            sample = outputs[i]
            detections = len(sample["instances"])
            dets = sample["instances"].get_fields()

            for det in range(detections):
                if not observations:
                    detection = {}
                    for key, val in dets.items():
                        detection[key] = val[det]
                    observations[obs_id] = [detection]

                else:
                    addThis = None
                    for (
                        group,
                        ds,
                    ) in observations.items():
                        for d in ds:
                            thisMask = dets["pred_masks"][det]
                            otherMask = d["pred_masks"]
                            overlap = torch.logical_and(thisMask, otherMask)
                            union = torch.logical_or(thisMask, otherMask)
                            IOU = overlap.sum() / float(union.sum())
                            if IOU <= iou_thres:
                                break
                            else:
                                detection = {}
                                for key, val in dets.items():
                                    detection[key] = val[det]
                                addThis = [group, detection]
                                break
                        if addThis:
                            break
                    if addThis:
                        observations[addThis[0]].append(addThis[1])
                    else:
                        obs_id += 1
                        detection = {}
                        for key, val in dets.items():
                            detection[key] = val[det]
                        observations[obs_id] = [detection]

        return observations

    def get_uncertainty(self, predictions, iterrations, height, width, mode="mean"):
        uncertainty_list = []

        device = "cuda"

        for key, val in predictions.items():

            mean_bbox = torch.mean(
                torch.stack([v["pred_boxes"].tensor for v in val]), axis=0
            )
            mean_mask = torch.mean(
                torch.stack(
                    [
                        v["pred_masks"].flatten().type(torch.cuda.FloatTensor)
                        for v in val
                    ]
                ),
                axis=0,
            )

            mean_mask[mean_mask < 0.25] = 0.0
            mean_mask = mean_mask.reshape(-1, height, width)
            mask_IOUs = []
            for v in val:
                current_mask = v["pred_masks"]
                overlap = torch.logical_and(mean_mask, current_mask)
                union = torch.logical_or(mean_mask, current_mask)
                if union.sum() > 0:
                    IOU = torch.divide(overlap.sum(), union.sum())
                    mask_IOUs.append(IOU.unsqueeze(0))

            if len(mask_IOUs) > 0:
                mask_IOUs = torch.cat(mask_IOUs)
            else:
                mask_IOUs = torch.tensor([float("NaN")]).to(device)

            bbox_IOUs = []
            mean_bbox = mean_bbox.squeeze(0)
            boxAArea = torch.multiply(
                (mean_bbox[2] - mean_bbox[0] + 1), (mean_bbox[3] - mean_bbox[1] + 1)
            )

            for v in val:
                current_bbox = v["pred_boxes"].tensor.squeeze(0)
                xA = torch.max(mean_bbox[0], current_bbox[0])
                yA = torch.max(mean_bbox[1], current_bbox[1])
                xB = torch.min(mean_bbox[2], current_bbox[2])
                yB = torch.min(mean_bbox[3], current_bbox[3])
                interArea = torch.multiply(
                    torch.max(torch.tensor(0).to(device), xB - xA + 1),
                    torch.max(torch.tensor(0).to(device), yB - yA + 1),
                )
                boxBArea = torch.multiply(
                    (current_bbox[2] - current_bbox[0] + 1),
                    (current_bbox[3] - current_bbox[1] + 1),
                )
                bbox_IOU = torch.divide(interArea, (boxAArea + boxBArea - interArea))
                bbox_IOUs.append(bbox_IOU.unsqueeze(0))

            if len(bbox_IOUs) > 0:
                bbox_IOUs = torch.cat(bbox_IOUs)
            else:
                bbox_IOUs = torch.tensor([float("NaN")]).to(device)

            val_len = torch.tensor(len(val)).to(device)
            outputs_len = torch.tensor(iterrations).to(device)

            u_spl_m = torch.clamp(torch.divide(mask_IOUs.sum(), val_len), min=0, max=1)
            u_spl_b = torch.clamp(torch.divide(bbox_IOUs.sum(), val_len), min=0, max=1)
            u_spl = torch.multiply(u_spl_m, u_spl_b)

            try:
                u_n = torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1)
            except:
                u_n = 0.0

            u_n = torch.multiply(u_n, u_n)
            u_h = torch.multiply(u_spl, u_n)
            if not torch.isnan(u_h.unsqueeze(0)) and u_spl != 1:
                uncertainty_list.append(u_h.unsqueeze(0))

        if uncertainty_list:
            uncertainty_list = torch.cat(uncertainty_list)

            if mode == "min":
                uncertainty = torch.min(uncertainty_list)
            elif mode == "mean":
                uncertainty = torch.mean(uncertainty_list)
            elif mode == "max":
                uncertainty = torch.max(uncertainty_list)
            else:
                uncertainty = torch.mean(uncertainty_list)

        else:
            uncertainty = torch.tensor([float("NaN")]).to(device)

        return uncertainty.detach().cpu().numpy().squeeze(0)
