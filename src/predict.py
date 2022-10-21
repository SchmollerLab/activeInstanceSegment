import numpy as np

from detectron2.engine import DefaultPredictor


def predict_image_in_acdc(cfg)

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))

    image_shape = masks[0].shape
    image = np.zeros(image_shape)

    for id in range(len(masks)):
        np.place(image, masks[id], id + 1)

    return image