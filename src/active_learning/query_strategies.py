
import sys
sys.path.append("..")

import random as rd
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from globals import *
from register_datasets import register_by_ids



class QueryStrategy(object):
    
    def __init__(self,cfg):
        
        self.cfg = cfg
        
    
    def sample(self,model, ids):
        pass
    
class RandomSampler(QueryStrategy):
    
    def sample(self,model, ids):
        num_samples = self.cfg.AL.INCREMENT_SIZE        
        samples = rd.sample(ids, num_samples)
        return samples

class GTknownSampler(QueryStrategy):
    
    def sample(self, model, ids):
        num_samples = self.cfg.AL.INCREMENT_SIZE
        
        id_pool = rd.sample(ids, min(600,len(ids)))
        
        register_by_ids("GTknownSampler_DS",id_pool, self.cfg.OUTPUT_DIR, self.cfg.AL.DATASETS.TRAIN_UNLABELED)

        
        evaluator = COCOEvaluator("GTknownSampler_DS", output_dir=self.cfg.OUTPUT_DIR)
        data_loader = build_detection_test_loader(self.cfg, "GTknownSampler_DS")
        inference_on_dataset(model, data_loader, evaluator)


        result_array = []
        image_ids = [image["image_id"] for image in DatasetCatalog.get("GTknownSampler_DS")]
        for image_id in image_ids:
            result = evaluator.evaluate(image_id)
            result_array.append(result)

        aps = np.array([result['segm']['AP'] for result in result_array])
        sample_ids = list(np.argsort(aps)[:num_samples])
        print("max aps: ", aps[sample_ids[0]])
        print("min aps: ", aps[list(np.argsort(aps)[:num_samples])[-1]])
        
        samples = [image_ids[id] for id in sample_ids]

        return samples