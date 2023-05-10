import os
from datetime import date

from src.globals import *
from utils.config_builder import get_config
from src.active_learning.al_trainer import ActiveLearningTrainer


def run_max_epochs():
    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)

    config_name = "mc_drop_al_hyp"

    for max_epochs in [100, 200, 300, 400]:
        cfg = get_config(config_name)
        cfg.SEED = 1337
        cfg.AL.MAX_TRAINING_EPOCHS = max_epochs
        cur_date = (
            "max_epochs_"
            + str(max_epochs)
            + "_"
            + str(date.today().month)
            + str(date.today().day)
        )
        for _ in range(3):
            cfg.SEED += 1
            al_trainer = ActiveLearningTrainer(
                cfg, cur_date=cur_date, is_test_mode=not running_on_server
            )
            al_trainer.run()
            al_trainer = None


def run_weight_init():
    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)

    config_name = "mc_drop_al_hyp"
    cfg = get_config(config_name)
    cfg.AL.MAX_LOOPS = 4
    cfg.AL.MAX_TRAINING_EPOCHS = 20
    for retrain in [False, True]:
        cfg.AL.RETRAIN = retrain
        cur_date = (
            "retrain_"
            + str(retrain)
            + "_"
            + str(date.today().month)
            + str(date.today().day)
        )
        for _ in range(3):
            cfg.SEED += 1
            al_trainer = ActiveLearningTrainer(
                cfg, cur_date=cur_date, is_test_mode=not running_on_server
            )
            al_trainer.run()
            al_trainer = None


def run_query_size():
    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)

    config_name = "mc_drop_al_hyp"
    cfg = get_config(config_name)
    cfg.AL.MAX_TRAINING_EPOCHS = 200

    for num_samples in [50, 100, 150]:
        cfg.SEED = 1337
        cfg.AL.MAX_TRAINING_EPOCHS = 100
        cfg.AL.INCREMENT_SIZE = num_samples
        cfg.AL.MAX_LOOPS = int(300 / num_samples + 1)
        cur_date = (
            "increment_size_"
            + str(num_samples)
            + "_"
            + str(date.today().month)
            + str(date.today().day)
        )
        for _ in range(3):
            cfg.SEED += 1
            al_trainer = ActiveLearningTrainer(
                cfg, cur_date=cur_date, is_test_mode=not running_on_server
            )
            al_trainer.run()
            al_trainer = None


if __name__ == "__main__":
    run_query_size()
