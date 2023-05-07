import os
from datetime import date

from src.globals import *
from utils.config_builder import get_config
from src.active_learning.al_trainer import ActiveLearningTrainer


if __name__ == "__main__":
    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)

    config_name = "mc_drop_al_hyp"
    cfg = get_config(config_name)

    for max_epochs in [100, 200, 300, 400]:
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
