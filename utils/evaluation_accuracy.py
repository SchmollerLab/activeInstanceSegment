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

from detectron2.evaluation.coco_evaluation import *
import json


class AccuracyEvaluator(COCOEvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            tasks=tasks,
            distributed=distributed,
            output_dir=output_dir,
            max_dets_per_image=max_dets_per_image,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas,
            allow_cached_coco=kpt_oks_sigmas,
        )

        self.ds_json = DatasetCatalog.get(dataset_name)

        self.object_pred_list = []

    def accuracy_eval_recall(self, image, outputs, height, width):
        im_id = image["image_id"]

        instances = list(filter(lambda x: x["image_id"] == im_id, self.ds_json))[0]

        instances = instances["annotations"]

        for j in range(len(instances)):
            instance = instances[j]

            obj_found = False
            for i in range(len(outputs["instances"])):
                output = outputs["instances"][i]
                mask = polygons_to_bitmask(
                    instance["segmentation"], height=height, width=width
                )
                pred = output.pred_masks.to("cpu").detach().numpy()
                iou = np.sum(mask & pred) / np.sum(mask | pred)

                if iou > 0.2:
                    self.object_pred_list.append(
                        {
                            "image_id": im_id,
                            "object_id": j,
                            "pred_mask": pred,
                            "iou": iou,
                            "pred_class": int(
                                output.pred_classes.to("cpu").detach().numpy()
                            ),
                            "true_class": instance["category_id"],
                            "detected": True,
                        }
                    )
                    obj_found = True
                    break
            if not obj_found:
                self.object_pred_list.append(
                    {
                        "image_id": im_id,
                        "object_id": j,
                        "detected": False,
                        "true_class": instance["category_id"],
                    }
                )

    def accuracy_eval(self, image, outputs, height, width):
        im_id = image["image_id"]

        instances = list(filter(lambda x: x["image_id"] == im_id, self.ds_json))[0]

        instances = instances["annotations"]

        for i in range(len(outputs["instances"])):
            output = outputs["instances"][i]

            obj_found = False

            for j in range(len(instances)):
                instance = instances[j]

                mask = polygons_to_bitmask(
                    instance["segmentation"], height=height, width=width
                )
                pred = output.pred_masks.to("cpu").detach().numpy()
                iou = np.sum(mask & pred) / np.sum(mask | pred)

                if iou > 0.2:
                    self.object_pred_list.append(
                        {
                            "image_id": im_id,
                            "object_id": j,
                            "pred_mask": pred,
                            "iou": iou,
                            "pred_class": int(
                                output.pred_classes.to("cpu").detach().numpy()
                            ),
                            "true_class": instance["category_id"],
                            "detected": True,
                        }
                    )
                    obj_found = True
                    break
            if not obj_found:
                self.object_pred_list.append(
                    {
                        "image_id": im_id,
                        "object_id": j,
                        "detected": False,
                        "true_class": instance["category_id"],
                    }
                )

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            print(input["image_id"])
            self.accuracy_eval(input, output, input["height"], input["width"])

    def evaluate_single_img(self, pred, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        predictions = pred

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def evaluate(self):
        return self.object_pred_list


if __name__ == "__main__":
    dataset = CHLAMY_DATASET
    test_dataset_name = dataset + "_test"
    config_name = "chlamy_random_al"

    model_path = (
        "/home/florian/GitRepos/activeCell-ACDC/output/chlamy_random_al/model_training"
    )

    register_datasets()

    wandb.init(
        project="activeCell-ACDC",
        name="",
        sync_tensorboard=True,
        mode="disabled",
    )

    default_cfg = get_config(config_name)
    cfg = default_cfg
    cfg.OUTPUT_DIR = "./al_output/classes_acdc_large_al"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    model = load_model(cfg, os.path.join(model_path, "best_model.pth"))

    evaluator = AccuracyEvaluator(test_dataset_name, cfg, False, output_dir="./")
    val_loader = build_detection_test_loader(cfg, test_dataset_name)

    # Use the created predicted model in the previous step
    res = inference_on_dataset(model, val_loader, evaluator)
