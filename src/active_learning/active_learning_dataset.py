import sys
sys.path.append("..")

import random as rd

from detectron2.data import MetadataCatalog, DatasetCatalog

from globals import *
from register_datasets import register_datasets, register_by_ids

class ActiveLearingDataset:
    
    def __init__(self, cfg):
        """
        
        
        """
        
        self.cfg = cfg
        
        register_datasets()

        # get ids of all images
        self.unlabeled_ids = [image["image_id"] for image in DatasetCatalog.get(cfg.AL.DATASETS.TRAIN_UNLABELED)]
        self.labeled_ids = []
        
        self.unlabeled_data_name = "temp_unlabeled_data_al"
        self.labeled_data_name = "temp_labeled_data_al"
        
        self.init_size = cfg.AL.INIT_SIZE
        self.increment_size = cfg.AL.INCREMENT_SIZE
        
        # set seed
        rd.seed(1337)
        sample_ids = rd.sample(self.unlabeled_ids, self.init_size)
        self.update_labeled_data(sample_ids)
        self.get_labeled_dataset()
        self.get_unlabled_dataset()
        
    
    def remove_data_from_catalog(self,name):
        
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        
        
    def get_labeled_dataset(self):
        self.remove_data_from_catalog(self.labeled_data_name)
        register_by_ids(self.cfg, self.labeled_data_name, self.labeled_ids)
        self.cfg.DATASETS.TRAIN = (self.labeled_data_name,)
    
    def get_unlabled_dataset(self):
        self.remove_data_from_catalog(self.unlabeled_data_name)
        register_by_ids(self.cfg, self.unlabeled_data_name,self.unlabeled_ids)
        self.cfg.AL.DATASETS.TRAIN_UNLABELED = self.unlabeled_data_name
    
    def update_labeled_data(self, sample_ids):
        print("update_labeled_data")
        # check if sample_ids are in unlabeled_ids
        if not (set(sample_ids) <= set(self.unlabeled_ids)):
            raise Exception("Some ids ({}) in sample_ids are not contained in unlabeled data pool: {}".format(len(list(set(sample_ids) - set(self.unlabeled_ids))),list(set(sample_ids) - set(self.unlabeled_ids))[:5])) 

        self.labeled_ids += sample_ids
        self.unlabeled_ids = list(set(self.unlabeled_ids) - set(sample_ids))
        
        self.get_labeled_dataset()
        self.get_unlabled_dataset()