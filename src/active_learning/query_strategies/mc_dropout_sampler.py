import numpy as np

from detectron2.data import DatasetCatalog

from src.globals import *
from utils.register_datasets import register_by_ids
from src.active_learning.query_strategies.query_strategy import UncertaintySampler


import numpy as np
import torch
from tqdm import tqdm
import wandb
import json


import torch
from itertools import chain
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import detectron2.data.transforms as T
from baal.bayesian.dropout import patch_module


class MCDropoutSampler(UncertaintySampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.strategy = "mc_dropout"
        self.clean_output_dir()

    def sample(self, cfg, ids, custom_model=None):
        """Sample datapoints using Monte Carlo dropout method

        Parameters
        ----------
        cfg
            Detectron2 config file
        ids
            id pool from which ids are sampled
        custom_model
            custom model which should be used for sampling. If not specified, model is built from config
        """

        num_samples = self.cfg.AL.INCREMENT_SIZE
        id_pool = self.presample_id_pool(cfg, ids, cfg.AL.SAMPLE_EVERY)
        register_by_ids(
            "ALSampler_DS",
            id_pool,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )

        if not custom_model:
            model = build_model(cfg)
            model = patch_module(model)
            model.eval()

            checkpointer = DetectionCheckpointer(model)
            print("loading default_model")
            checkpointer.load(os.path.join(cfg.OUTPUT_DIR, "last_model.pth"))
        else:
            print(f"loading custom model")
            model = custom_model

        ds_catalog = DatasetCatalog.get("ALSampler_DS")
        uncertainty_dict = {}
        print("running mc dropout sampling...")
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
                uncertainty_dict[im_json["image_id"]] = float(uncertainty)

        worst_ims = np.argsort(list(uncertainty_dict.values()))[-num_samples:]
        samples = [list(uncertainty_dict.keys())[id] for id in worst_ims]

        try:
            self.log_results(uncertainty_dict, samples)
        except:
            print("error while logging al results")

        self.counter += 1
        return samples

    def log_results(self, uncertainty_dict, samples):
        """Prints information about uncertainties and sampled datapoints to console and logs wandb"""

        with open(
            os.path.join(
                self.cfg.AL.OUTPUT_DIR,
                self.strategy,
                f"uncertainties{str(self.counter)}.json",
            ),
            "w",
        ) as file:
            json.dump(uncertainty_dict, file)

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
                "al": {
                    "min_uncertainty": min(list(uncertainty_dict.values())),
                    "mean_uncertainty": sum(list(uncertainty_dict.values()))
                    / len(list(uncertainty_dict.values())),
                    "max_uncertainty": max(list(uncertainty_dict.values())),
                }
            }
        )

        with open(
            os.path.join(
                self.cfg.AL.OUTPUT_DIR,
                self.strategy,
                f"{self.strategy}_samples{str(self.counter)}.txt",
            ),
            "w",
        ) as file:
            file.write("\n".join(samples))

    def get_samples(self, model, input_image, iterrations):
        """Perform inferences using Monte Carlo dropout

        Parameters
        ----------
        model
            model used for sampling
        input_image
            image as numpy array which should be usef for inferences
        iterrations
            number of inferences

        Returns
        -------
            list of predicted instances
        """

        with torch.no_grad():
            images, inputs = self.preprocess_image(input_image, model)

            features = model.backbone(images.tensor)

            proposals, box_features_pooler = self.get_backbone_roi_proposals(
                model, images, features
            )

            prediction_list = []
            for _ in range(iterrations):
                instances = self.get_instance_detections(
                    model, inputs, images, features, proposals, box_features_pooler
                )
                prediction_list.append(instances)

            return list(chain.from_iterable(prediction_list))


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
    query_strategy = MCDropoutSampler(cfg)

    cProfile.run("query_strategy.sample(cfg, al_dataset.unlabeled_ids)")
