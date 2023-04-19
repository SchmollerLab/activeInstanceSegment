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
 
class SingleImageCOCOEvaluator(COCOEvaluator):

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

        self.aps = {}



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
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                pred = [prediction]
                result = self.evaluate_single_img(pred, img_ids=[input["image_id"]])
                self.aps[input["image_id"]] = result


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
        return self.aps

    

    
results_path = "experiments/results"

dataset = ACDC_LARGE_CLS
config_name = "classes_acdc_large_al"

model_path = "/mnt/activeCell-ACDC/al_output/classes_acdc_large_al/random"
register_datasets()

test_data = DatasetCatalog.get("acdc_large_cls_test_slim")


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


for num_train_data in [15,30,45,60,75,90,105,120]: #[15, 240, 3000, 5800]
    
    model = load_model(
        cfg,
        os.path.join(model_path, f"best_model{str(num_train_data)}.pth"),
    )
    
    evaluator = SingleImageCOCOEvaluator("acdc_large_cls_test_slim", cfg, False,output_dir="./" )
    val_loader = build_detection_test_loader(cfg, "acdc_large_cls_test_slim")

    #Use the created predicted model in the previous step
    res = inference_on_dataset(model, val_loader, evaluator)
    
    with open(os.path.join(results_path,f"maps_{num_train_data}.txt"), "w") as fp:
        json.dump(res, fp)