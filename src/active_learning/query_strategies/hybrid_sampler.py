import os, sys
import random as rd
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from src.globals import *
from utils.register_datasets import register_by_ids, get_dataset_name
from src.active_learning.query_strategies.mc_dropout_sampler import MCDropoutSampler


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


import torch
from itertools import chain
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import detectron2.data.transforms as T
from baal.bayesian.dropout import patch_module


class HybridSampler(MCDropoutSampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.strategy = "hybrid"
        self.clean_output_dir()

    def sample(self, cfg, ids):
        num_samples = self.cfg.AL.INCREMENT_SIZE
        id_pool = self.presample_id_pool(cfg, ids, cfg.AL.SAMPLE_EVERY, random=True)

        if len(id_pool) <= num_samples:
            return id_pool

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
        samples_df = pd.DataFrame(
            data={
                "image_id": [],
                "uncertainty": [],
            }
        )
        feature_list = []
        sample_list = []

        print("running hybrid sampling...")
        for i in tqdm(range(len(ds_catalog))):
            with torch.no_grad():
                im_json = ds_catalog[i]
                im = self.load_image(im_json)

                instance_list = self.get_samples(model, im, cfg.AL.NUM_MC_SAMPLES)
                combinded_instances = self.get_combinded_instances(instance_list)

                height, width = im.shape[:2]
                uncertainty = self.get_uncertainty(
                    combinded_instances,
                    cfg.AL.NUM_MC_SAMPLES,
                    height,
                    width,
                    mode=cfg.AL.OBJECT_TO_IMG_AGG,
                )

                features = self.get_latent_feature(model, im)
                feature_list.append(features)

                sample_list.append(
                    {
                        "image_id": im_json["image_id"],
                        "uncertainty": float(uncertainty),
                    }
                )

        samples_df = pd.DataFrame.from_records(sample_list)
        samples_df["cluster"], samples_df["distance_center"] = self.get_k_means(
            feature_list, num_samples, samples_df["uncertainty"]
        )
        samples = []
        for cluster in samples_df.cluster.unique():
            df_tmp = samples_df[samples_df["cluster"] == cluster].copy()
            image_id = df_tmp[
                df_tmp["distance_center"] == df_tmp["distance_center"].min()
            ]["image_id"].values[0]
            samples.append(image_id)

        self.log_results(samples_df, samples)

        self.counter += 1
        return samples

    def log_results(self, samples_df, samples):
        samples_df.to_csv(
            os.path.join(
                self.cfg.AL.OUTPUT_DIR,
                self.strategy,
                f"sample_df{str(self.counter)}.csv",
            )
        )

        print("finished with hybrid sampling.")
        print("worst examples:", samples)

        with open(
            os.path.join(
                self.cfg.AL.OUTPUT_DIR,
                self.strategy,
                f"{self.strategy}_samples{str(self.counter)}.txt",
            ),
            "w",
        ) as file:
            file.write("\n".join(samples))

    def get_k_means(self, feature_list, n_clusters, sample_weight):
        np_feature_list = np.stack(feature_list)
        # umap_10d = UMAP(n_components=10, init="random", random_state=0)
        # proj_10d = umap_10d.fit_transform(np_feature_list)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
            X=np_feature_list, sample_weight=sample_weight
        )

        dist_transforms = kmeans.transform(np_feature_list)
        distances = dist_transforms[np.arange(dist_transforms.shape[0]), kmeans.labels_]
        return kmeans.labels_, distances

    def get_latent_feature(self, model, input_image, offs=10, layer="p5"):
        with torch.no_grad():
            images, _ = self.preprocess_image(input_image, model)
            features = model.backbone(images.tensor)

            mid_width = int(features[layer].shape[2] / 2)
            mid_height = int(features[layer].shape[3] / 2)
            feature_space = features[layer][
                0,
                :,
                mid_width - offs : mid_width + offs,
                mid_height - offs : mid_height + offs,
            ].flatten()

        return feature_space.detach().cpu().numpy()


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
    query_strategy = HybridSampler(cfg)
    print(query_strategy.sample(cfg, al_dataset.unlabeled_ids))
    # cProfile.run('query_strategy.sample(cfg, al_dataset.unlabeled_ids)')
