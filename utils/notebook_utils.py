from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


def load_model(cfg, model_path):
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)

    return model
