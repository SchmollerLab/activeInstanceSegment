import cv2
import numpy as np
import imutils

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.masks import polygons_to_bitmask

from utils.visualization.show_image import show_image


def load_model(cfg, model_path):
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)

    return model


def load_image(im_json):
    im = cv2.imread(im_json["file_name"])
    return im


def get_json_by_id(id, ds_json):
    return list(filter(lambda x: x["image_id"] == id, ds_json))[0]


def plot_ground_truth(image_json):
    w = image_json["width"]
    h = image_json["height"]

    np_image = np.zeros((h, w))
    cell_id = 1

    for obj in image_json["annotations"]:
        np_image = np.maximum(
            cell_id
            * polygons_to_bitmask(polygons=obj["segmentation"], height=h, width=w),
            np_image,
        )
        # cell_id += 1

    show_image(np_image, normalize=False)


def plot_ground_truth_object_det(image_json, cfg):
    in_image = cv2.imread(image_json["file_name"])
    visualizer = Visualizer(
        in_image[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=2,
    )
    out = visualizer.draw_dataset_dict(image_json)
    out_image = out.get_image()[:, :, ::-1]
    show_image(out_image)


def plot_ground_truth_rot(image_json, angle):
    w = image_json["width"]
    h = image_json["height"]

    np_image = np.zeros((h, w))
    cell_id = 1

    for obj in image_json["annotations"]:
        np_image = np.maximum(
            cell_id
            * polygons_to_bitmask(polygons=obj["segmentation"], height=h, width=w),
            np_image,
        )
        cell_id += 1

    np_image = imutils.rotate(np_image, angle=angle)
    show_image(np_image, normalize=False)
