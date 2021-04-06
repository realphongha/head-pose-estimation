import os
import torch
import inspect

from .utils.load_checkpoint import load_checkpoint
from .utils.functional import *
from .utils.preprocess import *
from .model.net2 import Net2


MODEL_CONFIG = {"net_type": Net2, "n_class": 3, "use_dist": False, "use_dist_angles": False,
                "n_points": 17}

CHECKPOINT = "assets/pretrained.pth"


def init_model(model_configs, checkpoint):
    model = model_configs["net_type"](**MODEL_CONFIG)
    model = load_checkpoint(model, checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def predict(model, landmarks):
    landmarks = torch.tensor(landmarks)
    if torch.cuda.is_available():
        landmarks = landmarks.cuda().float()
    landmarks = torch.unsqueeze(landmarks, 0) # add batch dimension
    outputs = model(landmarks)
    outputs = outputs.cpu().detach().numpy()
    return outputs


def to_angles(landmarks):
    if MODEL_CONFIG["n_points"] == 68:
        landmarks = landmarks
    elif MODEL_CONFIG["n_points"] in mapping_points:
        landmarks = mapping_points[MODEL_CONFIG["n_points"]](landmarks)
    else:
        print("[ERROR] %d-point landmarks is not supported" % MODEL_CONFIG["n_points"])
        quit()
    if MODEL_CONFIG["use_dist"]:
        landmarks = normalize_dist(points_to_dists(landmarks))
    elif MODEL_CONFIG["use_dist_angles"]:
        landmarks = landmarks_to_dist_angles(landmarks)
    else:
        landmarks = normalize_landmarks(standardize_landmarks(landmarks))

    checkpoint_abs_path = os.path.join(os.path.split(os.path.abspath(inspect.getfile(predict)))[0], CHECKPOINT)

    model = init_model(MODEL_CONFIG, checkpoint_abs_path)
    pose = predict(model, landmarks)[0]

    return pose


if __name__ == "__main__":
    pass
