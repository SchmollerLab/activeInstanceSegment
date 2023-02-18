import os, sys
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.append(PROJECT_ROOT)




import random as rd
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from src.globals import *
from src.register_datasets import register_by_ids, get_dataset_name
from src.active_learning.query_strategies import QueryStrategy


import numpy as np
import torch
import cv2
from tqdm import tqdm
import operator
import wandb
import json
import time
import pandas as pd
from sklearn.cluster import KMeans
from umap import UMAP


import torch
from itertools import chain
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import detectron2.data.transforms as T
from baal.bayesian.dropout import patch_module


class HybridSampler(QueryStrategy):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.strategy = "hybrid"
        self.clean_output_dir()
        self.max_entropy = self.calculate_max_entropy(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    def calculate_max_entropy(self, num_classes):
        least_confident = np.divide(np.ones(num_classes), num_classes).astype(np.float32)
        probs = torch.from_numpy(least_confident)
        max_entropy = torch.distributions.Categorical(probs).entropy()
        return max_entropy

    def sample(self, cfg, ids):

        num_samples = self.cfg.AL.INCREMENT_SIZE

        if cfg.AL.SAMPLE_EVERY <= 1:
            id_pool = ids
        else:
            rand_int = rd.randint(0,cfg.AL.SAMPLE_EVERY)
            id_pool = list(filter(lambda x: (int(x.split("_")[-1]) + rand_int) % cfg.AL.SAMPLE_EVERY == 0, ids))

        register_by_ids(
            "ALSampler_DS",
            id_pool,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )

        model = build_model(cfg)
        model = patch_module(model)
        model.eval()

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))

        ds_catalog = DatasetCatalog.get("ALSampler_DS")
        samples_df = pd.DataFrame(data={
            "image_id":[],
            "uncertainty":[],
        })
        feature_list = []
        
        layer = "p4"
        offs = 10
        
        print("running mc dropout sampling...")
        for i in tqdm(range(len(ds_catalog))):

            im_json = ds_catalog[i]
            im = cv2.imread(im_json["file_name"])
            predictions, features = self.get_mc_dropout_samples(cfg, model, im, cfg.AL.NUM_MC_SAMPLES)
            mid_width = int(features[layer].shape[2]/2)
            mid_height = int(features[layer].shape[3]/2)
            feature_space = features[layer][0,:,mid_width-offs:mid_width+offs,mid_height-offs:mid_height+offs].flatten()
            feature_list.append(feature_space.detach().cpu().numpy())
            outputs = self.get_observations(predictions)
            height, width = im.shape[:2]
            uncertainty = self.get_uncertainty(outputs, cfg.AL.NUM_MC_SAMPLES, height, width, mode=cfg.AL.OBJECT_TO_IMG_AGG)

            samples_df = samples_df.append({
                "image_id": im_json["image_id"],
                "uncertainty":float(uncertainty),
            }, ignore_index=True)
           
        np_feature_list = np.stack(feature_list)
        
        umap_10d = UMAP(n_components=10, init='random', random_state=0)
        proj_10d = umap_10d.fit_transform(np_feature_list)
        kmeans = KMeans(n_clusters=cfg.AL.INCREMENT_SIZE, random_state=0, n_init="auto").fit(proj_10d)
        samples_df["cluster"] = kmeans.labels_
        
        samples = []
        for cluster in samples_df.cluster.unique():
            df_tmp = samples_df[samples_df["cluster"] == cluster].copy()
            image_id = samples_df[samples_df["uncertainty"] == df_tmp["uncertainty"].max()]["image_id"].values[0]
            samples.append(image_id)
        
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
                softmaxes = torch.index_select(predictions[0],0,pred_inds[0])
                
                pred_instances = model.roi_heads.forward_with_given_boxes(
                    features, pred_instances
                )
                
                outputs = model._postprocess(pred_instances, inputs, images.image_sizes)
                for output in outputs:
                    output["instances"].set("softmaxes",softmaxes)
                prediction_list.append(outputs)
            return list(chain.from_iterable(prediction_list)), features

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

    def get_semantic_uncertainty(self, val, device = "cuda"):

        torch_softmax = torch.nn.Softmax(dim=0)

        softmaxes = torch.stack([torch_softmax(v['softmaxes']) for v in val])
        if len(softmaxes[0]) == 1:
             u_sem = torch.ones(1).to(device)
        else:           
            mean_softmaxes = torch.mean(softmaxes, axis = 0)
            u_sem = torch.max(mean_softmaxes)

        return u_sem

    def get_mask_uncertainty(self, val, height, width, val_len, device="cuda"):

        
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

        u_spl_m = torch.clamp(torch.divide(mask_IOUs.sum(), val_len), min=0, max=1)

        return u_spl_m



    def get_box_uncertainty(self, val, val_len, device="cuda"):
        mean_bbox = torch.mean(
            torch.stack([v["pred_boxes"].tensor for v in val]), axis=0
        )

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

        u_spl_b = torch.clamp(torch.divide(bbox_IOUs.sum(), val_len), min=0, max=1)

        return u_spl_b

    def get_detection_uncertainty(self, iterrations, val_len, device="cuda"):
        try:
            outputs_len = torch.tensor(iterrations).to(device)
            u_n = torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1)
        except:
            u_n = 0.0

        return u_n

    def get_uncertainty(self, predictions, iterrations, height, width, mode="max"):
        uncertainty_list = []

        device = "cuda"

        for key, val in predictions.items():

            val_len = torch.tensor(len(val)).to(device)
         
            u_sem = self.get_semantic_uncertainty(val=val, device=device)

            u_spl_m = self.get_mask_uncertainty(val=val, height=height, width=width, val_len=val_len, device=device)
            u_spl_b = self.get_box_uncertainty(val=val, val_len=val_len, device=device)
            
            u_spl = torch.multiply(u_spl_m, u_spl_b)

            if u_sem > 0:
                u_sem_spl = torch.multiply(u_sem, u_spl)
            else:
                u_sem_spl = u_spl

            u_n = self.get_detection_uncertainty(iterrations=iterrations, val_len=val_len, device=device)
           
            u_h = torch.multiply(u_sem_spl, u_n)

            # transform certainty to uncertainty
            u_h = 1 - u_h

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
                uncertainty = torch.max(uncertainty_list)
                
        else:
            uncertainty = torch.tensor([float("NaN")]).to(device)

        return uncertainty.detach().cpu().numpy().squeeze(0)
