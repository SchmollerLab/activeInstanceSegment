import os
import math
import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from skimage import exposure, io

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def show_image(ims, normalize=True):
    # figure(figsize=(10, 10), dpi=80)
    if not isinstance(ims, list):
        if normalize:
            im_cont = exposure.equalize_adapthist(ims)
        else:
            im_cont = ims
        plt.imshow(im_cont)

    else:

        fig = plt.figure(figsize=(15, 10))

        num_figures = len(ims)
        cols = 3
        rows = int(math.ceil(num_figures / cols))

        for i in range(num_figures):
            if normalize:
                im_cont = exposure.equalize_adapthist(ims[i])
            else:
                im_cont = ims[i]
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(im_cont)

    plt.show()


def plot_prediction(image_json, dataset_name, cfg):
    
    logger = setup_logger(output="./log/main.log")
    logger.setLevel(0)

    logger = setup_logger(output="./log/main.log",name="null_logger") 
    logger.addHandler(logging.NullHandler())
    logging.getLogger('detectron2').setLevel(logging.WARNING)
    logging.getLogger('detectron2').addHandler(logging.NullHandler())

    # ground truth
    raw_im = cv2.imread(image_json["file_name"])

    visualizer = Visualizer(raw_im[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=2)
    out = visualizer.draw_dataset_dict(image_json)
    ground_truth_im = out.get_image()[:, :, ::-1]
    
    # prediction
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(raw_im)


    v = Visualizer(raw_im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    predicted_instances = np.asarray(outputs["instances"].pred_masks.to("cpu"))
    predicted_mask = np.zeros(predicted_instances[0].shape)

    for id in range(len(predicted_instances)):
        np.place(predicted_mask, predicted_instances[id], 1)

    predicted_im = out.get_image()[:, :, ::-1]


    visualizer = Visualizer(np.zeros(raw_im[:, :, ::-1].shape), metadata=MetadataCatalog.get(dataset_name), scale=2)
    new_im_json = image_json.copy()
    new_im_json["annotations"] = []
    for anno in image_json["annotations"]:
        new_anno = anno.copy()
        new_anno["bbox"] = [0,0,0,0]
        new_im_json["annotations"].append(new_anno)
    out = visualizer.draw_dataset_dict(new_im_json)
    ground_truth_mask = (np.array(out.get_image()[:, :, ::-1]) != 0).max(axis=2).astype(np.uint8)
    ground_truth_mask = cv2.resize(ground_truth_mask*255, dsize=(predicted_mask.shape[1],predicted_mask.shape[0]), interpolation=cv2.INTER_CUBIC)


    show_image([raw_im, ground_truth_im,predicted_im, ground_truth_mask, predicted_mask, (ground_truth_mask > 0) - predicted_mask])
