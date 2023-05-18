import wandb
import torch
import os
import logging
import pandas as pd
from statistics import mode

from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog

from src.globals import *
from utils.register_datasets import register_datasets
from utils.config_builder import get_config
from utils.notebook_utils import *

from src.active_learning.al_trainer import *
from src.active_learning.mc_dropout_sampler import *
from src.active_learning.tta_sampler import *


def val2dicts(val):
    preds = []
    for v in val:
        pred_masks = v["pred_masks"].detach().cpu().numpy()
        pred_classes = v["pred_classes"].detach().cpu().numpy()

        preds.append(
            {
                "pred_masks": pred_masks,
                "pred_classes": pred_classes,
            }
        )

    return preds


def get_uncertainties(cfg, im_json, model, query_strategy):
    im = query_strategy.load_image(im_json)
    instance_list = query_strategy.get_samples(model, im, cfg.AL.NUM_MC_SAMPLES)
    combinded_instances = query_strategy.get_combinded_instances(instance_list)

    height, width = im.shape[:2]

    uncertainties = []

    for key, val in combinded_instances.items():
        val_len = torch.tensor(len(val)).to("cuda")

        if query_strategy.cfg.MODEL.ROI_HEADS.NUM_CLASSES > 1:
            u_sem = (
                query_strategy.get_semantic_certainty(val, device="cuda")
                .detach()
                .cpu()
                .numpy()
            )
        else:
            u_sem = 0
        u_mask = (
            query_strategy.get_mask_certainty(
                val, height, width, val_len, device="cuda"
            )
            .detach()
            .cpu()
            .numpy()
        )
        u_box = (
            query_strategy.get_box_certainty(val, val_len, device="cuda")
            .detach()
            .cpu()
            .numpy()
        )
        u_det = (
            query_strategy.get_detection_certainty(
                cfg.AL.NUM_MC_SAMPLES, val_len, device="cuda"
            )
            .detach()
            .cpu()
            .numpy()
        )

        cpu_val = val2dicts(val)

        uncertainties.append(
            {
                "val": cpu_val,
                "u_sem": u_sem,
                "u_mask": u_mask,
                "u_box": u_box,
                "u_det": u_det,
            }
        )

    return uncertainties


def samples_to_preds(samples):
    predicted_mask = (
        np.mean(np.stack([val["pred_masks"] for val in samples["val"]]), axis=0) > 0.25
    )

    unique, counts = np.unique(
        [val["pred_classes"] for val in samples["val"]], return_counts=True
    )
    index = np.argmax(counts)
    predicted_class = unique[index]
    return predicted_mask, predicted_class


def evaluate_uncertainties(cfg, datasetname, model, stragtegy):
    # load dataset
    register_datasets()
    data_set = DatasetCatalog.get(datasetname)

    predicted_uncertainties = []

    for im_json in tqdm(data_set):
        instances = im_json["annotations"]
        uncertainties = get_uncertainties(cfg, im_json, model, stragtegy)

        for j in range(len(instances)):
            obj_found = False
            for i in range(len(uncertainties)):
                gt_mask = polygons_to_bitmask(
                    instances[j]["segmentation"],
                    height=im_json["height"],
                    width=im_json["width"],
                )

                pred_mask, pred_class = samples_to_preds(uncertainties[i])

                iou = np.sum(gt_mask & pred_mask) / np.sum(gt_mask | pred_mask)

                if iou > 0.2:
                    predicted_uncertainties.append(
                        {
                            "image_id": im_json["image_id"],
                            "object_id": j,
                            "pred_mask": pred_mask,
                            "iou": iou,
                            "pred_class": pred_class,
                            "true_class": instances[j]["category_id"],
                            "detected": True,
                            "detection_type": "tp",
                            "u_sem": uncertainties[i]["u_sem"],
                            "u_mask": uncertainties[i]["u_mask"],
                            "u_det": uncertainties[i]["u_det"],
                        }
                    )
                    obj_found = True

                    uncertainties.pop(i)
                    break

            if not obj_found:
                predicted_uncertainties.append(
                    {
                        "image_id": im_json["image_id"],
                        "object_id": j,
                        "detected": False,
                        "true_class": instances[j]["category_id"],
                        "detection_type": "fn",
                    }
                )
        for i in range(len(uncertainties)):
            pred_mask, pred_class = samples_to_preds(uncertainties[i])
            predicted_uncertainties.append(
                {
                    "image_id": im_json["image_id"],
                    "object_id": i,
                    "pred_mask": pred_mask,
                    "pred_class": pred_class,
                    "detected": False,
                    "detection_type": "fp",
                    "u_sem": uncertainties[i]["u_sem"],
                    "u_mask": uncertainties[i]["u_mask"],
                    "u_det": uncertainties[i]["u_det"],
                }
            )

        for uncertainty in uncertainties:
            pred_mask, pred_class = samples_to_preds(uncertainty)

    return predicted_uncertainties


if __name__ == "__main__":
    logger = setup_logger(output="./log/main.log", name="null_logger")
    logger.addHandler(logging.NullHandler())
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("detectron2").addHandler(logging.NullHandler())

    datasetname = ACDC_LARGE_CLS + "_test_slim"
    config_name = "final_random_al"

    default_cfg = get_config(config_name)
    cfg = default_cfg
    cfg.OUTPUT_DIR = "./al_output/classes_acdc_large_al"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    model_path = "/home/florian/GitRepos/activeCell-ACDC/output"
    model = load_model(cfg, os.path.join(model_path, "best_model.pth"))

    stragtegy = MCDropoutSampler(cfg)

    print(evaluate_uncertainties(cfg, datasetname, model, stragtegy))
