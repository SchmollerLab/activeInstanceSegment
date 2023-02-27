import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


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
