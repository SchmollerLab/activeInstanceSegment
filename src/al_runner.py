import os
from datetime import date
from globals import *
from config_builder import get_config
from active_learning.al_trainer import ActiveLearningTrainer


if __name__ == "__main__":
    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)
    cfg = get_config("acdc_large_al")

    cur_date = str(date.today().month) + str(date.today().day)
    for _ in range(2):
        cfg.SEED += 1
        al_trainer = ActiveLearningTrainer(cfg, is_test_mode=not running_on_server)
        #al_trainer.run(dataset=ACDC_LARGE, query_strat=MC_DROPOUT, cur_date=cur_date)
        al_trainer.run(dataset=ACDC_LARGE, query_strat=RANDOM, cur_date=cur_date)
