import torch, detectron2
import wandb
import os
import shutil
import sys

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_writers
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.append(PROJECT_ROOT)

from src.test import do_test
from src.globals import *
from src.logging.wandb_event_writer import WandBWriter



def clean_output_dir(output_dir):
    try:
        shutil.rmtree(output_dir)
    except:
        pass

    os.mkdir(output_dir)
    
    


def do_train(cfg, logger, resume=False):

    if not resume:
        clean_output_dir(cfg.OUTPUT_DIR)

    model = build_model(cfg)
    model.train()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    # define counter for early stopping
    early_counter = 0
    max_ap = 0
    max_result = {}
    max_early_counter = 0

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )

    max_iter = cfg.SOLVER.MAX_ITER

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) + [WandBWriter()] if comm.is_main_process() else []
    )

    # define augmentations
    augs = [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.ResizeShortestEdge(
            short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
            max_size=cfg.INPUT.MAX_SIZE_TRAIN,
            sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        ),
    ]
    data_loader = build_detection_train_loader(
        cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs)
    )

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                
                res = do_test(cfg, model=model, logger=logger)
                
                comm.synchronize()

                wandb.log(
                    {"early_stopping_ap": (res["segm"]["AP"] + res["bbox"]["AP"]) / 2}
                )
                if (res["segm"]["AP"] + res["bbox"]["AP"]) / 2 < max_ap:
                    early_counter += 1
                    max_early_counter = max(max_early_counter, early_counter)

                    print(
                        "new ap:",
                        (res["segm"]["AP"] + res["bbox"]["AP"]) / 2,
                        "max_ap",
                        max_ap,
                        "add counter: ",
                        early_counter,
                    )
                    if early_counter >= cfg.EARLY_STOPPING_ROUNDS:
                        print("stopping training")
                        break
                else:
                    print(
                        "new ap:",
                        (res["segm"]["AP"] + res["bbox"]["AP"]) / 2,
                        "max_ap",
                        max_ap,
                        "adjust max",
                    )
                    max_result = res
                    max_ap = max(max_ap, (res["segm"]["AP"] + res["bbox"]["AP"]) / 2)
                    checkpointer.save("best_model")
                    early_counter = 0

                wandb.log({"max_early_counter": max_early_counter})

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):

                for writer in writers:
                    writer.write()

    return max_result
