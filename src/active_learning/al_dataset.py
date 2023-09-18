import sys
import os

import random as rd

from detectron2.data import MetadataCatalog, DatasetCatalog

from src.globals import *
from utils.register_datasets import register_datasets, register_by_ids


class ActiveLearingDataset:
    def __init__(self, cfg):
        """ """

        self.cfg = cfg

        register_datasets()

        # get ids of all images
        self.unlabeled_jsons = DatasetCatalog.get(cfg.AL.DATASETS.TRAIN_UNLABELED)
        self.unlabeled_ids = [
            image["image_id"] for image in self.unlabeled_jsons if image["file_name"]
        ]

        self.labeled_ids = []
        self.unlabeled_data_name = "temp_unlabeled_data_al"
        self.labeled_data_name = "temp_labeled_data_al"

        self.init_size = cfg.AL.INIT_SIZE
        self.increment_size = cfg.AL.INCREMENT_SIZE

        # set seed
        rd.seed(cfg.SEED)

        sample_ids = rd.sample(self.unlabeled_ids, self.init_size)
        self.update_labeled_data(sample_ids)
        self.register_labeled_dataset()
        self.register_unlabled_dataset()


    def get_len_labeled(self):
        """Returns length of labeled dataset"""
        return len(self.labeled_ids)


    def get_len_unlabeled(self):
        """Something

        Parameters
        ----------
        param1
            desc1
        param1
            desc2
        """
        return len(self.unlabeled_ids)


    def get_num_objects(self):
        """Returns number of labeled objects"""
        labeled_ds_jsons = DatasetCatalog.get(self.labeled_data_name)
        num_objects = sum(
            [len(json_img["annotations"]) for json_img in labeled_ds_jsons]
        )
        return num_objects


    def remove_data_from_catalog(self, name):
        """Remove dataset from data catalog"""
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)


    def register_labeled_dataset(self):
        """Register labeled dataset in Detectron2 datasets"""
        self.remove_data_from_catalog(self.labeled_data_name)
        register_by_ids(
            self.labeled_data_name,
            self.labeled_ids,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )
        self.cfg.DATASETS.TRAIN = (self.labeled_data_name,)


    def register_unlabled_dataset(self):
        """Register unlabeled dataset in Detectron2 datasets"""
        self.remove_data_from_catalog(self.unlabeled_data_name)
        register_by_ids(
            self.unlabeled_data_name,
            self.unlabeled_ids,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )


    def update_labeled_data(self, sample_ids):
        """Adds new ids to labeled dataset and removes them from unlabeled dataset

        Parameters
        ----------
        sample_ids
            ids which should be added to labeled dataset
        """
        print("update_labeled_data")
        # check if sample_ids are in unlabeled_ids
        if not (set(sample_ids) <= set(self.unlabeled_ids)):
            raise Exception(
                "Some ids ({}) in sample_ids are not contained in unlabeled data pool: {}".format(
                    len(list(set(sample_ids) - set(self.unlabeled_ids))),
                    list(set(sample_ids) - set(self.unlabeled_ids))[:5],
                )
            )

        self.labeled_ids += sample_ids
        self.unlabeled_ids = list(set(self.unlabeled_ids) - set(sample_ids))

        self.register_labeled_dataset()
        self.register_unlabled_dataset()


if __name__ == "__main__":
    from utils.config_builder import get_config

    cfg = get_config("acdc_large_al")
    cfg.AL.INIT_SIZE = 5
    al_ds = ActiveLearingDataset(cfg=cfg)
    print("num objs", al_ds.get_num_objects())
    print(
        len(al_ds.unlabeled_ids),
        len(DatasetCatalog.get(cfg.AL.DATASETS.TRAIN_UNLABELED)),
    )
    print(len(al_ds.labeled_ids))
    al_ds.update_labeled_data(al_ds.unlabeled_ids[1:3])
    print("num objs", al_ds.get_num_objects())
    print(
        len(al_ds.unlabeled_ids),
        len(DatasetCatalog.get(cfg.AL.DATASETS.TRAIN_UNLABELED)),
    )
    print(len(al_ds.labeled_ids))
