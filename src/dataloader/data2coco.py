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
 
import datetime
import json
import re
import fnmatch
import pycococreatortools
 
from typing import Callable

import imageio
 
ROOT_PATH = "/vol/volume/data" #"./data/"
IMAGE_DIR_NAME = "images"
ANNOTATION_DIR_NAME = "annotations"
TEST = "test"
TRAIN = "train"
 
INFO = {
    "description": "cellpose data in COCO format",
    "url": "",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "Florian Bridges",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}
 
LICENSES = [{"id": 1, "name": "", "url": ""}]
 
CATEGORIES = [
    {
        "id": 0,
        "name": "cell",
        "supercategory": "cell",
    }
]
 
class Data2cocoConverter:

    def __init__(self, root_dir, dataset_name, data_splits=[TEST,TRAIN]) -> None:
        
        self.dataset_name = dataset_name
        self.data_splits = data_splits
        self.raw_images_path = os.path.join(root_dir,"raw_data",dataset_name)
        self.save_images_path = os.path.join(root_dir, dataset_name)

        self.create_dir_to_path(root_dir,dataset_name)
        for data_split in data_splits:

            self.create_dir_to_path(self.save_images_path,data_split)        
            self.create_dir_to_path(self.save_images_path,data_split + "/annotations")        
            self.create_dir_to_path(self.save_images_path,data_split + "/images")        



    def create_dir_to_path(self, path, dir_name):
        if not os.path.exists(os.path.join(path, dir_name)):
            os.mkdir(os.path.join(path, dir_name))


    def convert(self):
        print("converting {} images ...".format(self.dataset_name))
        
        # prepare images
        self.iterate_images()
        image_id = 1
        segmentation_id = 1
    
        for split_type in self.data_splits:
            self.convert_data_to_coco(
                image_id, segmentation_id, os.path.join(self.save_images_path, split_type)
            )


    def iterate_images(self):
        pass
 
    def augment_store_image(self, full_id, image, mask, split_type):
    
        # flip_combs: [[horizontal,vertical]]
        flip_combs = [[False,False],[True,False],[False,True],[True,True]] 
    
        for flip_comb in flip_combs:
    
            full_id_with_flip = full_id + self.get_flip_str(flip_comb)
        
            image_flipped = self.flip_image(image, flip_comb)
            mask_full = self.flip_image(mask, flip_comb)
    
            # save masks independently
            self.store_image(full_id_with_flip, image_flipped, mask_full, split_type)
               
 
    def store_image(self, id, image, mask, split_type):

        # save image as png
        plt.imsave(
            os.path.join(self.save_images_path, split_type, "images", id + ".png"),
            image,
            cmap="gray",
        )
    
        # save masks independently
        labels = np.unique(mask)
        for label in labels[1:]:
            mask_single_cell: np.uint8 = (mask == label).astype(np.uint8)
            plt.imsave(
                os.path.join(
                    self.save_images_path,
                    split_type,
                    "annotations",
                    id + "_cell_" + str(label) + ".png",
                ),
                mask_single_cell,
                cmap="gray",
            ) 
    
    def flip_image(self, image, flip_type):
    
        for dim in [0,1]:
            if flip_type[dim]:
                image = np.flip(image,dim)
    
        return image
    
    def get_flip_str(self, flip_type):
    
        flip_str = "H" + str(flip_type[0])[0] + "V" + str(flip_type[1])[0]
        return flip_str
    
            
    
    def filter_for_png(self, root, files):
        file_types = ["*.png"]
        file_types = r"|".join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
    
        return files
    
    def filter_for_annotations(self, root, files, image_filename):
        file_types = ["*.png"]
        file_types = r"|".join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + "_"
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [
            f
            for f in files
            if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])
        ]
    
        return files
    
    def convert_data_to_coco(self, image_id, segmentation_id, path_dir):
    
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": [],
        }
    
        for root, _, files in os.walk(os.path.join(path_dir, IMAGE_DIR_NAME)):
            image_files = self.filter_for_png(root, files)
            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size
                )
                coco_output["images"].append(image_info)
    
                # filter for associated png annotations
                for root, _, files in os.walk(os.path.join(path_dir, ANNOTATION_DIR_NAME)):
                    annotation_files = self.filter_for_annotations(root, files, image_filename)
    
                    
                    # go through each associated annotation
                    for annotation_filename in annotation_files:
    
                        class_id = [
                            x["id"] for x in CATEGORIES if x["name"] in annotation_filename
                        ][0]
    
                        category_info = {
                            "id": class_id,
                            "is_crowd": "crowd" in image_filename,
                        }
                        binary_mask = np.asarray(
                            Image.open(annotation_filename).convert("1")
                        ).astype(np.uint8)
    
                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id,
                            image_id,
                            category_info,
                            binary_mask,
                            image.size,
                            tolerance=0,
                        )
    
                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)
    
                        segmentation_id = segmentation_id + 1
    
                image_id = image_id + 1
            
        with open("{}/cell_acdc_coco_ds.json".format(path_dir), "w") as output_json_file:
            json.dump(coco_output, output_json_file)

class Cellpose2cocoConverter(Data2cocoConverter):

    def __init__(self, root_dir) -> None:

        dataset_name = "cellpose"
        super().__init__(root_dir, dataset_name)
        self.raw_images_path = os.path.join(self.raw_images_path,"data")

    
    def iterate_images(self):
        
        num_images = 540
        num_train = math.ceil(num_images * 0.8)

        ids = list(range(num_images))
        rd.shuffle(ids)
        ids_dict = {}
        ids_dict[TRAIN] = ids[:num_train]
        ids_dict[TEST] = ids[num_train:]
    
        for split_type in [TRAIN,TEST]:
            for id in tqdm(ids_dict[split_type]):
                id_str = ("000" + str(id))[-3:]
                image = mpimg.imread(os.path.join(self.raw_images_path, id_str + "_img.png"))
                mask = mpimg.imread(os.path.join(self.raw_images_path, id_str + "_masks.png"))
                self.augment_store_image(str(id),image,mask, split_type)

class SmallACDC2cocoConverter(Data2cocoConverter):


    def __init__(self, root_dir) -> None:

        dataset_name = "acdc_small"
        super().__init__(root_dir, dataset_name)
        self.raw_images_path = os.path.join(self.raw_images_path,"TimeLapse_2D")
    
    def iterate_images(self):

        id: int = 0
        split_type: str = TEST
        directory = os.fsencode(self.raw_images_path)
        for obj in os.listdir(directory):
            name = os.fsdecode(obj)
            if name.find("_labeled") != -1:
                for pos in os.listdir(os.fsencode(self.raw_images_path + "/" + name)):
                    pos_name = os.fsdecode(pos)
                    for file in os.listdir(
                        os.fsencode(self.raw_images_path + "/" + name + "/" + pos_name + "/Images")
                    ):
                        filename = os.fsdecode(file)
                        if filename.find("_phase_contr.tif") != -1:
                            base_filename = filename.replace("phase_contr.tif", "")
                    data = self.load_video(name, pos_name, base_filename)
                    self.store_images(data, str(id), split_type)
                    id += 1

                    split_type = TRAIN

    def store_images(self, data: np.array, id: int, split_type: str = "train") -> None:

        images: np.array = data[0]
        masks: np.array = data[1]
        
        num_images: int = images.shape[0]
        for i in range(num_images):
            self.augment_store_image(str(id) + "_" + str(i),images[i],masks[i], split_type)

    def load_video(self, experiment_name, position, filename):
        path = self.raw_images_path + "/" + experiment_name + "/" + position + "/Images/"
        print("loading:", path)

        vid = io.imread(path + filename + "phase_contr.tif")
        print("loaded video with shape:\t", vid.shape)

        lables = np.load(path + filename + "segm.npz")["arr_0"]
        print("loaded lables with shape:\t", lables.shape)

        data = np.array([vid, lables])

        return data
 
class LargeACDC2cocoConverter(Data2cocoConverter):


    def __init__(self, root_dir) -> None:

        dataset_name = "acdc_large"
        super().__init__(root_dir, dataset_name)
    
    def iterate_images(self):
        paths = []
        phase_contr_tifs = []
        phase_contr_npzs = []
        segms = []
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
                            if filename.find("Ph3_aligned.npz") != -1 ^ "phase_contr_aligned.npz" != -1:
                                print(filename)
                                phase_contr_npz = filename
                            if filename.find("phase_contr.tif")  != -1 ^ filename.find("Ph3.tif") != -1:
                                phase_contr_tif = filename
                            if filename.find("segm.npz") != -1:
                                segm = filename

                        paths.append(self.raw_images_path + "/" + acdc_ds_name  + "/" + experiment_name + "/" + position_name + "/Images")
                        phase_contr_npzs.append(phase_contr_npz)
                        phase_contr_tifs.append(phase_contr_tif)
                        segms.append(segm)

        df = pd.DataFrame(data={"paths": paths, "phc_npz": phase_contr_npzs, "phc_tif": phase_contr_tifs, "segm": segms})
        df.to_csv("found_files.csv")



    def store_images(self, data: np.array, id: int, split_type: str = "train") -> None:

        images: np.array = data[0]
        masks: np.array = data[1]

        num_images: int = min(images.shape[0], masks.shape[0])

        for i in range(num_images):
            if (masks[i] > 0).sum():
                self.store_image(str(id) + "_" + str(i),images[i],masks[i], split_type)
                break

    def load_video(self, folder_name, experiment_name, position, filename, segm_filename):
        path = self.raw_images_path + "/" + folder_name + "/" + experiment_name + "/" + position + "/Images/"
        print("loading:", path)

        try:
            vid = np.load(path + filename.replace("phase_contr.tif","phase_contr_aligned.npz"))["arr_0"]
        except:
            print("aligned.npz not found")
            vid = io.imread(path + filename)

        lables = np.load(path + segm_filename)["arr_0"]
        

        print("loaded video with shape:\t", vid.shape)
        print("loaded lables with shape:\t", lables.shape)

        data = np.array([vid, lables], dtype="object")

        return data
if __name__ == "__main__":

    #cpc = Cellpose2cocoConverter(ROOT_PATH)
    #cpc.convert()
    #small_acdc_conv = SmallACDC2cocoConverter(ROOT_PATH)
    #small_acdc_conv.convert()
    large_acdc_conv = LargeACDC2cocoConverter(ROOT_PATH)
    large_acdc_conv.convert()
