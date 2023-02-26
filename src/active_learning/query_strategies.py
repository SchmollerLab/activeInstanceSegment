import os, sys

import random as rd
import numpy as np
import torch
import cv2

import shutil

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import detectron2.data.transforms as T

from src.globals import *
from utils.register_datasets import register_by_ids


class QueryStrategy(object):
    def __init__(self, cfg):

        self.cfg = cfg
        self.counter = 1

    def clean_output_dir(self):
        try:
            shutil.rmtree(os.path.join(self.cfg.AL.OUTPUT_DIR, self.strategy))
        except:
            pass
        
        try:
            os.mkdir("./al_output")
        except:
            pass

        try:
            os.mkdir(self.cfg.AL.OUTPUT_DIR)
        except:
            pass

        os.mkdir(os.path.join(self.cfg.AL.OUTPUT_DIR, self.strategy))

    def sample(self, cfg, ids):
        pass


class RandomSampler(QueryStrategy):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.strategy = "random"
        self.clean_output_dir()

    def sample(self, cfg, ids):
        num_samples = self.cfg.AL.INCREMENT_SIZE
        rd.seed(cfg.SEED)
        samples = rd.sample(ids, num_samples)
        
        with open(os.path.join(self.cfg.AL.OUTPUT_DIR, self.strategy, f"{self.strategy}_samples{str(self.counter)}.txt"),"w") as file:
            file.write("\n".join(samples))
        self.counter += 1
        return samples

class UncertaintySampler(QueryStrategy):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.max_entropy = self.calculate_max_entropy(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.classification = cfg.MODEL.ROI_HEADS.NUM_CLASSES > 1

    def calculate_max_entropy(self, num_classes):
        least_confident = np.divide(np.ones(num_classes), num_classes).astype(np.float32)
        probs = torch.from_numpy(least_confident)
        max_entropy = torch.distributions.Categorical(probs).entropy()
        return max_entropy

    def presample_id_pool(self, cfg, ids):

        if cfg.AL.SAMPLE_EVERY <= 1:
            id_pool = ids
        else:
            rand_int = rd.randint(0,cfg.AL.SAMPLE_EVERY)
            id_pool = list(filter(lambda x: (int(x.split("_")[-1]) + rand_int) % cfg.AL.SAMPLE_EVERY == 0, ids))
        
        return id_pool

    def load_image(self, im_json):
        im = cv2.imread(im_json["file_name"])

        return im


    def preprocess_image(self, input_image, model):

        height, width = input_image.shape[:2]
        image = self.aug.get_transform(input_image).apply_image(input_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        images = model.preprocess_image(inputs)

        return images, inputs

    def get_backbone_roi_proposals(self, model, images, features):

        proposals, _ = model.proposal_generator(images, features, None)
        features_ = [features[f] for f in model.roi_heads.box_in_features]

        box_features_pooler = model.roi_heads.box_pooler(
            features_, [x.proposal_boxes for x in proposals]
        )

        return proposals, box_features_pooler


    def get_instance_detections(self, model, inputs, images, features, proposals, box_features_pooler):
        box_features = model.roi_heads.box_head(box_features_pooler)
        predictions = model.roi_heads.box_predictor(box_features)

        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(
            predictions, proposals
        )
        
        
        pred_instances = model.roi_heads.forward_with_given_boxes(
            features, pred_instances
        )
        
        instances = model._postprocess(pred_instances, inputs, images.image_sizes)

        if self.classification:
            softmaxes = torch.index_select(predictions[0],0,pred_inds[0])
            for output in instances:
                output["instances"].set("softmaxes",softmaxes)

        return instances

    def get_combinded_instances(self, outputs, iou_thres=0.2):
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
                if torch.sum(dets["pred_masks"][det]) < 50:
                    continue

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

        outputs_len = torch.tensor(iterrations).to(device)
        #u_n = torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1)
        u_n = 2*torch.abs(torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1) - 0.5)


        return u_n

    def get_uncertainty(self, predictions, iterrations, height, width, mode="max", bbox=True):
        uncertainty_list = []

        device = "cuda"

        for key, val in predictions.items():

            val_len = torch.tensor(len(val)).to(device)

            u_spl_m = self.get_mask_uncertainty(val=val, height=height, width=width, val_len=val_len, device=device)
            if bbox:
                u_spl_b = self.get_box_uncertainty(val=val, val_len=val_len, device=device)
                u_spl = torch.multiply(u_spl_m, u_spl_b)
            else:
                u_spl = u_spl_m

            if self.classification:
                u_sem = self.get_semantic_uncertainty(val=val, device=device)
                u_sem_spl = torch.multiply(u_sem, u_spl)
            else:
                u_sem_spl = u_spl

            u_n = self.get_detection_uncertainty(iterrations=iterrations, val_len=val_len, device=device)
           
            #u_h = torch.multiply(u_sem_spl, u_n)
            u_h = u_n
            # transform certainty to uncertainty
            u_h = 1 - u_h

            if not torch.isnan(u_h.unsqueeze(0)): # and u_spl != 1:
                uncertainty_list.append(u_h.unsqueeze(0))

        if uncertainty_list:
            uncertainty_list = torch.cat(uncertainty_list)
            if mode == "min":
                uncertainty = torch.min(uncertainty_list)
            elif mode == "mean":
                uncertainty = torch.mean(uncertainty_list)
            elif mode == "max":
                uncertainty = torch.max(uncertainty_list)
            elif mode == "sum":
                uncertainty = torch.sum(uncertainty_list)
            else:
                uncertainty = torch.max(uncertainty_list)
                
        else:
            uncertainty = torch.tensor([float("NaN")]).to(device)

        return uncertainty.detach().cpu().numpy().squeeze(0)



class GTknownSampler(QueryStrategy):
    def sample(self, model, ids):
        num_samples = self.cfg.AL.INCREMENT_SIZE

        id_pool = rd.sample(ids, min(600, len(ids)))

        register_by_ids(
            "GTknownSampler_DS",
            id_pool,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )

        evaluator = COCOEvaluator("GTknownSampler_DS", output_dir=self.cfg.OUTPUT_DIR)
        data_loader = build_detection_test_loader(self.cfg, "GTknownSampler_DS")
        inference_on_dataset(model, data_loader, evaluator)

        result_array = []
        image_ids = [
            image["image_id"] for image in DatasetCatalog.get("GTknownSampler_DS")
        ]
        for image_id in image_ids:
            result = evaluator.evaluate(image_id)
            result_array.append(result)

        aps = np.array([result["segm"]["AP"] for result in result_array])
        sample_ids = list(np.argsort(aps)[:num_samples])
        print("max aps: ", aps[sample_ids[0]])
        print("min aps: ", aps[list(np.argsort(aps)[:num_samples])[-1]])

        samples = [image_ids[id] for id in sample_ids]

        return samples
