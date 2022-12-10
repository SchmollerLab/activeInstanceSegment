import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd
from tqdm import tqdm
import time
 
import datetime
import json
import re
import fnmatch
import pycococreatortools

import sys
sys.path.insert(0,sys.path[0] + "/..") 
from globals import *

from data2coco import Data2cocoConverter

class LargeACDC2cocoConverter(Data2cocoConverter):


    def __init__(self, root_dir, filename) -> None:

        dataset_name = "acdc_large"
        self.filename = filename
        super().__init__(root_dir, dataset_name, data_splits=["train", "test_1", "test_2"])
    
    def iterate_images(self):
        
        df_clean = pd.read_csv(os.path.join(self.raw_images_path,self.filename))

        index = df_clean.index.to_numpy() 
        np.random.seed(1221)
        np.random.shuffle(index)
        data_dict = {}
        data_dict["train"] = index[:33]
        data_dict["test_1"] = index[33:38]
        data_dict["test_2"] = index[38:]

        for data_split in self.data_splits:
            print("loading data for split {}:".format(data_split))
            for id in tqdm(data_dict[data_split]):
                row = df_clean.iloc[id]

                data = self.load_video(path=row["paths"], phc_npz=row["phc_npz"], phc_tif=row["phc_tif"], segm=row["segm"])

                image_id = row["paths"].replace(self.raw_images_path + "/","").replace("/Images","").replace("Position","pos").replace("/","_")
                self.store_images(id=image_id, data=data, split_type=data_split, max_image=row["max_image"])



    def store_images(self, data, id, split_type = "train", max_image=-1) -> None:

        images: np.array = data[0]
        masks: np.array = data[1]

        if max_image < 0:
            max_image = 1e6
        num_images: int = min(images.shape[0], masks.shape[0], max_image)

        
        for i in range(num_images):
            if (masks[i] > 0).sum():
                self.store_image(str(id) + "_" + str(i),images[i],masks[i], split_type)

    def load_video(self, path, phc_npz, phc_tif, segm):

        if phc_npz != "":
            if phc_npz.find(".npz") != -1:
                vid = np.load(os.path.join(path, phc_npz))["arr_0"]
            else:
                vid = np.load(os.path.join(path, phc_npz))
        else:
            vid = io.imread(os.path.join(path, phc_tif))

        masks =  np.load(os.path.join(path, segm))["arr_0"]
        data = np.array([vid, masks], dtype="object")

        return data




    def build_data_map(self):

        paths = []
        phase_contr_tifs = []
        phase_contr_npzs = []
        segms = []
        base_dict = os.fsencode(self.raw_images_path)
        for acdc_ds in os.listdir(base_dict):
            acdc_ds_name = os.fsdecode(acdc_ds)
            if acdc_ds_name.find(".zip") == -1 and acdc_ds_name.find(".csv") == -1:
                experiment_dict = os.fsencode(os.path.join(self.raw_images_path,acdc_ds_name))
                for experiment in os.listdir(experiment_dict):
                    experiment_name = os.fsdecode(experiment)
                    for position in os.listdir(os.fsencode(self.raw_images_path + "/" + acdc_ds_name + "/" + experiment_name)):
                        position_name = os.fsdecode(position)
                        
                        phase_contr_npz = ""
                        phase_contr_tif = ""
                        segm = ""

                        for file in os.listdir(
                            os.fsencode(self.raw_images_path + "/" + acdc_ds_name  + "/" + experiment_name + "/" + position_name + "/Images")
                        ):
                            filename = os.fsdecode(file)
                            if (filename.find("Ph3_aligned.np") != -1) or (filename.find("phase_contr_aligned.np") != -1):
                                phase_contr_npz = filename
                            if (filename.find("phase_contr.tif")  != -1) or (filename.find("Ph3.tif") != -1):
                                phase_contr_tif = filename
                            if filename.find("segm.npz") != -1:
                                segm = filename

                        paths.append(self.raw_images_path + "/" + acdc_ds_name  + "/" + experiment_name + "/" + position_name + "/Images")
                        phase_contr_npzs.append(phase_contr_npz)
                        phase_contr_tifs.append(phase_contr_tif)
                        segms.append(segm)

        df = pd.DataFrame(data={"paths": paths, "phc_npz": phase_contr_npzs, "phc_tif": phase_contr_tifs, "segm": segms})
        df_clean = df[df["segm"] != ""].copy().reset_index()
        df_clean["min_image"] = 0
        df_clean["max_image"] = -1
        df_clean.to_csv(os.path.join(self.raw_images_path,"data_map-" + str(int(time.time())) + ".csv"))
        

if __name__ == "__main__":

    large_acdc_conv = LargeACDC2cocoConverter(BASE_DATA_PATH, filename="data_map-1670164283.csv")
    #large_acdc_conv.build_data_map()
    large_acdc_conv.convert()