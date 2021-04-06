import numpy as np


def normalize_landmarks(landmarks):
    landmarks = landmarks.flatten()
    return landmarks/np.linalg.norm(landmarks)


def normalize_dist(dist):
    return dist/np.linalg.norm(dist)


def standardize_landmarks(landmarks):
    return landmarks - np.min(landmarks, axis=0)
