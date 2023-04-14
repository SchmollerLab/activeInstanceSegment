import wandb
import torch
import os
import logging
import pandas as pd


from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog

from src.globals import *
from utils.register_datasets import register_datasets
from utils.config_builder import get_config
from utils.notebook_utils import *

from src.active_learning.al_trainer import *
from src.active_learning.mc_dropout_sampler import *
from src.active_learning.tta_sampler import *


logger = setup_logger(output="./log/main.log", name="null_logger")
logger.addHandler(logging.NullHandler())
logging.getLogger("detectron2").setLevel(logging.WARNING)
logging.getLogger("detectron2").addHandler(logging.NullHandler())


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


def get_uncertainties(im_json, model, query_strategy):
    im = query_strategy.load_image(im_json)
    instance_list = query_strategy.get_samples(model, im, cfg.AL.NUM_MC_SAMPLES)
    combinded_instances = query_strategy.get_combinded_instances(instance_list)

    height, width = im.shape[:2]
    agg_uncertainty = query_strategy.get_uncertainty(
        combinded_instances,
        cfg.AL.NUM_MC_SAMPLES,
        height,
        width,
        mode=cfg.AL.OBJECT_TO_IMG_AGG,
    )

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

    return uncertainties, agg_uncertainty


results_path = "experiments/results/mc_drop_known_vs_unknown"
if not os.path.exists(results_path):
    os.makedirs(results_path)


dataset = ACDC_LARGE_CLS
config_name = "classes_acdc_large_al"

model_path = (
    "/home/florian/GitRepos/activeCell-ACDC/output/final_random_al/al_output/random"
)
register_datasets()

test_data = DatasetCatalog.get("acdc_large_cls_test_slim")
train_data = DatasetCatalog.get("acdc_large_cls_train")


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

rd.seed(1337)
sub_samples = rd.sample(test_data, 50)

is_train_dir = {image_json["image_id"]: 0 for image_json in sub_samples}


id_path = "/home/florian/GitRepos/activeCell-ACDC/output/final_random_al/al_output/random/small_train_id.txt"

with open(id_path, "r") as file:
    ids = file.read().split("\n")


train_jsons = rd.sample(
    [image_json for image_json in train_data if image_json["image_id"] in ids], 50
)
is_train_dir.update({image_json["image_id"]: 1 for image_json in train_jsons})

sub_samples += train_jsons


for num_train_data in [200]:
    for num_mc_samples in [10]:
        cfg_test = cfg
        cfg_test.AL.NUM_MC_SAMPLES = num_mc_samples
        mc_strategy = MCDropoutSampler(cfg_test)

        for dropout_prob in [0.35]:
            cfg_test = cfg

            cfg_test.MODEL.ROI_HEADS.DROPOUT_PROBABILITY = dropout_prob
            cfg_test.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = dropout_prob
            cfg_test.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = dropout_prob

            cfg_test = cfg
            cfg_test.AL.NUM_MC_SAMPLES
            mc_model = load_model(
                cfg_test,
                os.path.join(model_path, f"last_model{str(num_train_data)}.pth"),
            )
            mc_model = patch_module(mc_model)

            print(
                "num_mc_samples",
                num_mc_samples,
                "dropout_prob",
                dropout_prob,
                "model_train_size",
                num_train_data,
            )

            records = []
            for im_json in tqdm(sub_samples):
                single_im_unc = []
                for run_id in range(1):
                    uncertainties, agg_uncertainty = get_uncertainties(
                        im_json, mc_model, mc_strategy
                    )
                    records.append(
                        {
                            "num_mc_samples": num_mc_samples,
                            "dropout_prob": dropout_prob,
                            "model_train_size": num_train_data,
                            "image_id": im_json["image_id"],
                            "run_id": run_id,
                            "agg_uncertainty": agg_uncertainty,
                            "is_train": is_train_dir[im_json["image_id"]]
                            # "uncertainties": uncertainties,
                        }
                    )

            df = pd.DataFrame.from_records(records)
            df.to_csv(
                os.path.join(
                    results_path,
                    f"results_{num_mc_samples}_{dropout_prob}_{num_train_data}.csv",
                )
            )
