import os
from datetime import date
from argparse import ArgumentParser

from src.globals import *
from utils.config_builder import get_config
from src.active_learning.al_trainer import ActiveLearningTrainer


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config_filename",
        help="Path to pipeline configuration",
        metavar="FILE",
    )

    args = parser.parse_args()
    config_filename = args.config_filename

    running_on_server = os.getenv("IS_SERVER") == "true"
    print("running on server:", running_on_server)

    config_name = config_filename.split("/")[-1].replace(".yaml", "")
    cfg = get_config(config_name, complete_path=config_filename)
    
    cur_date = "u_det_kmeans" + str(date.today().month) + str(date.today().day)
    for _ in range(2):

        cfg.SEED += 1
        al_trainer = ActiveLearningTrainer(cfg, cur_date=cur_date, is_test_mode=not running_on_server)
        al_trainer.run()
        al_trainer = None
