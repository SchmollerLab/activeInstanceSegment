from query_strategies import QueryStrategy

import os
import sys
sys.path.append("..")

import random as rd
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from globals import *
from register_datasets import register_by_ids


import numpy as np
import torch
import cv2
from tqdm import tqdm
import operator
from nn_modules.montecarlo_dropout import *
from nn_modules.observations import *
from nn_modules.uncertainty import *

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


def uncertainty_pooling(pool_list, pool_size, cfg, config, max_entropy, mcd_iterations, mode):
    pool = {}
    cfg.MODEL.ROI_HEADS.SOFTMAXES = True
    predictor = MonteCarloDropoutHead(cfg, mcd_iterations)
    device = cfg.MODEL.DEVICE

    if len(pool_list) > 0:
        ## find the images from the pool_list the algorithm is most uncertain about
        for d in tqdm(range(len(pool_list))):
            filename = pool_list[d]
            if os.path.isfile(os.path.join(config['traindir'], filename)):
                img = cv2.imread(os.path.join(config['traindir'], filename))
                width, height = img.shape[:-1]
                outputs = predictor(img)

                obs = observations(outputs, config['iou_thres'])
                img_uncertainty = uncertainty(obs, mcd_iterations, max_entropy, width, height, device, mode) ## reduce the iterations when facing a "CUDA out of memory" error

                if not np.isnan(img_uncertainty):
                    if len(pool) < pool_size:
                        pool[filename] = float(img_uncertainty)
                    else:
                        max_id, max_val = max(pool.items(), key=operator.itemgetter(1))
                        if float(img_uncertainty) < max_val:
                            del pool[max_id]
                            pool[filename] = float(img_uncertainty)

        sorted_pool = sorted(pool.items(), key=operator.itemgetter(1))
        pool = {}
        for k, v in sorted_pool:
            pool[k] = v
    else:
        print("All images are used for the training, stopping the program...")

    return pool
