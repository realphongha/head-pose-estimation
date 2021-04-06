import numpy as np
import scipy.io as sio


def conv_98p_to_68p(l98):
    # converts 98 points landmarks annotation to 68 points type
    return np.array(l98[:33:2] + l98[33:38] + l98[42:47] + l98[51:60] +
                    l98[60:62] + l98[63:66] + [l98[67]] + l98[68:70] + l98[71:74] + l98[75:96])


def read_mat_annotation(mat_file):
    # gets euler angles and 2D landmarks from .mat annotation file
    mat = sio.loadmat(mat_file)
    pt = mat["pt2d"]
    landmarks = np.array(list(zip(pt[0], pt[1])))
    pitch, yaw, roll = map(lambda x: x * 180 / np.pi, mat['Pose_Para'][0][:3])
    x_min = min(pt[0, :])
    y_min = min(pt[1, :])
    x_max = max(pt[0, :])
    y_max = max(pt[1, :])
    return landmarks, (pitch, yaw, roll), (x_min, y_min, x_max, y_max)


def landmarks_to_facebox(landmarks, get_int=True):
    pt_x = [mark[0] for mark in landmarks]
    pt_y = [mark[1] for mark in landmarks]
    if get_int:
        return int(min(pt_x)), int(min(pt_y)), int(max(pt_x)), int(max(pt_y))
    return min(pt_x), min(pt_y), max(pt_x), max(pt_y)
