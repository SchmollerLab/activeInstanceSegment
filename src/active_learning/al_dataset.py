import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

import random as rd

from detectron2.data import MetadataCatalog, DatasetCatalog

from src.globals import *
from src.register_datasets import register_datasets, register_by_ids


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
        # self.dict_aug_ids_by_id = self.precompute_augmentation_ids(self.unlabeled_ids)
        self.labeled_ids = []
        # self.labeled_ids_aug = []

        self.unlabeled_data_name = "temp_unlabeled_data_al"
        self.labeled_data_name = "temp_labeled_data_al"

        self.init_size = cfg.AL.INIT_SIZE
        self.increment_size = cfg.AL.INCREMENT_SIZE

        # set seed
        rd.seed(cfg.SEED)
        
        sample_ids = rd.sample(self.unlabeled_ids, self.init_size)
        self.update_labeled_data(sample_ids)
        self.get_labeled_dataset()
        self.get_unlabled_dataset()

    def get_len_labeled(self):
        return len(self.labeled_ids)

    def get_num_objects(self):
        labeled_ds_jsons = DatasetCatalog.get(self.labeled_data_name)
        num_objects = sum([len(json_img["annotations"]) for json_img in labeled_ds_jsons])
        return num_objects
        

    def precompute_augmentation_ids(self, unlabeled_ids):

        dict_jsons_by_filename = {
            record["file_name"]: record for record in self.unlabeled_jsons
        }
        dict_jsons_by_id = {
            record["image_id"]: record for record in self.unlabeled_jsons
        }
        dict_aug_ids_by_id = {}
        for id in unlabeled_ids:
            record = dict_jsons_by_id[id]
            file_name_cut = record["file_name"].split("H")[0]
            ids = []
            for h in ["T", "F"]:
                for v in ["T", "F"]:
                    file_name = file_name_cut + "H" + h + "V" + v + ".png"
                    aug_record = dict_jsons_by_filename[file_name]
                    ids.append(aug_record["image_id"])
            dict_aug_ids_by_id[record["image_id"]] = ids

        return dict_aug_ids_by_id

    def remove_data_from_catalog(self, name):

        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    def get_labeled_dataset(self):
        self.remove_data_from_catalog(self.labeled_data_name)
        register_by_ids(
            self.labeled_data_name,
            self.labeled_ids,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )
        self.cfg.DATASETS.TRAIN = (self.labeled_data_name,)

    def get_unlabled_dataset(self):
        self.remove_data_from_catalog(self.unlabeled_data_name)
        register_by_ids(
            self.unlabeled_data_name,
            self.unlabeled_ids,
            self.cfg.OUTPUT_DIR,
            self.cfg.AL.DATASETS.TRAIN_UNLABELED,
        )

    def update_labled_ids_aug(self):
        for id in self.labeled_ids:
            self.labeled_ids_aug += self.dict_aug_ids_by_id[id]

    def update_labeled_data(self, sample_ids):
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

        # self.update_labled_ids_aug()
        self.get_labeled_dataset()
        self.get_unlabled_dataset()


if __name__ == "__main__":

    from src.config_builder import get_config

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
