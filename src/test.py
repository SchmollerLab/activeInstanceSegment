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
    num_ds = len(cfg.DATASETS.TEST)
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        for metric in results_i.keys():
            if metric in results.keys():
                results[metric] += results_i[metric]/num_ds
            else:
                results[metric] = results_i[metric]/num_ds
        
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    
    logger.info(results)
    wandb.log(results)
    model.train()
    return results
