import torch, detectron2
from collections import OrderedDict
import wandb

import detectron2.utils.comm as comm
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.evaluation import print_csv_format


def do_test(cfg, model=None, logger=None):

    if model is None:
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    else:
        results_sum = {}
        num_ds = len(list(results.keys()))
        for dataset_name in results.keys():
            for metric in results[dataset_name].keys():
                if metric in results_sum.keys():
                    results_sum[metric] += results[dataset_name][metric]/num_ds
                else:
                    results_sum[metric] = results[dataset_name][metric]/num_ds

        results = results_sum
    logger.info(results)
    wandb.log(results)
    model.train()
    return results
