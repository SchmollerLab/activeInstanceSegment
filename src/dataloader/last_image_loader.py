
import os
import numpy as np
import pandas as pd
from skimage import exposure, io
from tqdm import tqdm
 
import datetime
import json
import re
import fnmatch
import pycococreatortools
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd

from data2coco import Data2cocoConverter

import sys
sys.path.insert(0,sys.path[0] + "/..") 
from globals import *

class lastACDC2cocoConverter(Data2cocoConverter):


    def __init__(self, root_dir) -> None:

        dataset_name = "acdc_last_images"
        super().__init__(root_dir, dataset_name, data_splits=["test"])
    
    def iterate_images(self):
        paths = []
        phase_contr_tifs = []
        phase_contr_npzs = []
        segms = []
        self.raw_images_path = self.raw_images_path.replace("acdc_last_images", "acdc_large")
        base_dict = os.fsencode(self.raw_images_path)
        for acdc_ds in os.listdir(base_dict):
            acdc_ds_name = os.fsdecode(acdc_ds)
            if acdc_ds_name.find(".zip") == -1:
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

        index = df_clean.index.to_numpy() 
        np.random.seed(1337)
        np.random.shuffle(index)
        data_dict = {}
        data_dict["test"] = index[:]

        for data_split in self.data_splits:
            print("loading data for split {}:".format(data_split))
            for id in tqdm(data_dict[data_split]):
                row = df_clean.iloc[id]

                data = self.load_video(path=row["paths"], phc_npz=row["phc_npz"], phc_tif=row["phc_tif"], segm=row["segm"])

                image_id = row["paths"].replace(self.raw_images_path + "/","").replace("/Images","").replace("Position","pos").replace("/","_")
                self.store_images(id=image_id, data=data, split_type=data_split)



    def store_images(self, data, id, split_type = "train") -> None:

        images: np.array = data[0]
        masks: np.array = data[1]

        num_images: int = min(images.shape[0], masks.shape[0])

        i = num_images-1
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
if __name__ == "__main__":

    large_acdc_conv = lastACDC2cocoConverter(BASE_DATA_PATH)
    large_acdc_conv.convert()