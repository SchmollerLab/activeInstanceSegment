import os

from globals import *
from config_builder import get_config
from active_learning.active_learning_trainer import ActiveLearningTrainer



if __name__ == "__main__":
    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)
    cfg = get_config("acdc_small_al_20")
    al_trainer = ActiveLearningTrainer(cfg, is_test_mode=not running_on_server)
    al_trainer.run(dataset=ACDC_SMALL, query_strat=MC_DROPOUT)
    
