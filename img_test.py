import math
import os
import cv2
import numpy as np

from pathlib import Path
from math import pi, atan2, asin, cos, sin

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from utils.annotation import read_mat_annotation, conv_98p_to_68p
from utils.functional import r_vec_to_euler


def get_landmarks_in_pair(l_marks):
    return list(zip(l_marks[::2], l_marks[1::2]))


def mae(y_test, y_pred):
    return np.sum(np.abs(y_test-y_pred))/y_test.shape[0]


def main1():
    root_path = "/mnt/hdd10tb/Datasets/FaceDirection/WFLW"
    label_path = os.path.join(root_path, 'WFLW_annotations/list_98pt_test/list_98pt_test.txt')
    img_path = os.path.join(root_path, 'WFLW_images')
    with open(label_path) as f:
        db = f.read().splitlines()
    n_samples = 15
    for i in range(n_samples):
        sample = db[i]
        raw_landmarks = list(map(float, sample.split()[:-1]))
        landmarks = conv_98p_to_68p(get_landmarks_in_pair(raw_landmarks))
        img_filename = sample.split()[-1]
        path_to_img = os.path.join(img_path, img_filename)
        img = cv2.imread(path_to_img)
        height, width = img.shape[:2]
        pose_estimator = PoseEstimator(img_size=(height, width))
        mark_detector = MarkDetector()
        mark_detector.draw_marks(img, landmarks, color=(0, 255, 0))
        # cv2.imwrite(os.path.join("output", "%s-mark.jpg" % Path(path_to_img).stem), img)
        pose = pose_estimator.solve_pose_by_68_points_img(landmarks, return_euler=False)
        p, y, r = pose_estimator.r_vec_to_euler(pose[0])
        # pose_estimator.draw_annotation_box(img, pose[0], pose[1], color=(255, 128, 128))
        pose_estimator.draw_axes(img, pose[0], pose[1])
        # cv2.imwrite(os.path.join("output", "%s-pose.jpg" % Path(path_to_img).stem), img)
        # landmarks[33] is the nose point
        PoseEstimator.draw_axes_euler(img, y, p, r, tdx=landmarks[33][0], tdy=landmarks[33][1],
                        x_color=(0, 0, 100), y_color=(0, 100, 0), z_color=(100, 0, 0))
        cv2.imwrite(os.path.join("output", "%s-ypr-pose.jpg" % Path(path_to_img).stem), img)


def main2():
    root_path = r"E:\Workspace\data\face-direction\300W_LP\AFW"
    list_file = list(filter(lambda x: x[-6:] == "_0.jpg", sorted(os.listdir(root_path))))
    n_samples = len(list_file)
    sum_mae = 0.0
    for file in list_file[:n_samples]:
        img_path = os.path.join(root_path, file)
        mat_path = os.path.join(root_path, file.replace("jpg", "mat"))
        print(mat_path)
        landmarks, euler_angles, face_box = read_mat_annotation(mat_path)
        pitch, yaw, roll = euler_angles
        img = cv2.imread(img_path)
        # cv2.imwrite(os.path.join("output", "%s-original.jpg" % Path(img_path).stem), img)
        height, width = img.shape[:2]
        pose_estimator = PoseEstimator(img_size=(height, width))
        pose = pose_estimator.solve_pose_by_68_points(landmarks, return_euler=False)
        p, y, r = pose_estimator.r_vec_to_euler(pose[0])
        MarkDetector.draw_marks(img, landmarks, color=(0, 255, 0))
        # pose_estimator.draw_axes(img, pose[0], pose[1])
        # pose_estimator.draw_annotation_box(img, pose[0], pose[1], color=(255, 128, 128))
        print("ground truth:", (pitch, yaw, roll), "calculated:", (p, y, r))
        mae_ = mae(np.array([pitch, yaw, roll]), np.array([p, y, r]))
        print("mae:", mae_)
        sum_mae += mae_
        PoseEstimator.draw_axes_euler(img, y, p, r, tdx=landmarks[33][0], tdy=landmarks[33][1],
                        x_color=(0, 0, 100), y_color=(0, 100, 0), z_color=(100, 0, 0))
        PoseEstimator.draw_axes_euler(img, yaw, pitch, roll, tdx=landmarks[10][0], tdy=landmarks[10][1])
        cv2.imwrite(os.path.join("output", "%s-pose.jpg" % Path(img_path).stem), img)
    print("Final MAE: %.3f for %d sample(s)" % (sum_mae/n_samples, n_samples))


if __name__ == "__main__":
    main2()