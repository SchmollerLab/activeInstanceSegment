import torch, detectron2
import wandb
import yaml
import os

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

try:
    from test import do_test
except:
    from src.test import do_test


def do_train(cfg, logger, resume=False):

    model = build_model(cfg)
    model.train()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # wandb.config.update(yaml.load(cfg.dump()))

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    # define counter for early stopping
    early_counter = 0
    max_ap = 0

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    
    # define augmentations
    augs = [
        T.RandomFlip(prob=0.5,horizontal=True,vertical=False),
        T.RandomFlip(prob=0.5,horizontal=False,vertical=True)
    ]
    data_loader = build_detection_train_loader(cfg,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))
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
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

                wandb.log(
                    {"early_stopping_ap": (res["segm"]["AP"] + res["bbox"]["AP"]) / 2}
                )
                if (res["segm"]["AP"] + res["bbox"]["AP"]) / 2 < max_ap:
                    early_counter += 1
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
                    max_ap = max(max_ap, (res["segm"]["AP"] + res["bbox"]["AP"]) / 2)
                    checkpointer.save("best_model")
                    early_counter = 0

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):

                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
    return cfg
