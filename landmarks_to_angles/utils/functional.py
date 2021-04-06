import math
import numpy as np
import scipy.io as sio


# yaw, pitch, roll -> W, Z, Y, X
def euler2quat(yaw, pitch, roll):
    roll = roll * math.pi / 180
    pitch = pitch * math.pi / 180
    yaw = yaw * math.pi / 180

    x = math.sin(pitch / 2) * math.sin(yaw / 2) * math.cos(roll / 2) + math.cos(pitch / 2) * math.cos(
        yaw / 2) * math.sin(roll / 2)
    y = math.sin(pitch / 2) * math.cos(yaw / 2) * math.cos(roll / 2) + math.cos(pitch / 2) * math.sin(
        yaw / 2) * math.sin(roll / 2)
    z = math.cos(pitch / 2) * math.sin(yaw / 2) * math.cos(roll / 2) - math.sin(pitch / 2) * math.cos(
        yaw / 2) * math.sin(roll / 2)
    w = math.cos(pitch / 2) * math.cos(yaw / 2) * math.cos(roll / 2) - math.sin(pitch / 2) * math.sin(
        yaw / 2) * math.sin(roll / 2)
    return [w, x, y, z]


def quat2euler(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + y * x), 1 - 2 * (z * z + y * y))

    roll = roll * 180 / math.pi
    pitch = pitch * 180 / math.pi
    yaw = yaw * 180 / math.pi

    return [yaw, pitch, roll]


def read_mat(mat_path, pt3d=False):
    # Get facebox, 2D landmarks and Euler angles from .mat files
    mat = sio.loadmat(mat_path)
    if pt3d:
        pt = mat['pt3d_68']
    else:
        pt = mat['pt2d']

    landmarks = np.array(list(zip(pt[0], pt[1])))

    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose = pre_pose_params[:3]

    x_min = min(pt[0, :])
    y_min = min(pt[1, :])
    x_max = max(pt[0, :])
    y_max = max(pt[1, :])

    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi

    return np.array(landmarks), np.array((yaw, pitch, roll)), (x_min, y_min, x_max, y_max)


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def points_to_dists(landmarks):
    n = len(landmarks)
    dists = [landmarks[0][0], landmarks[0][1],
             landmarks[1][0], landmarks[1][1],
             landmarks[2][0], landmarks[2][1]]
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            dists.append(euclidean_dist(landmarks[i],  landmarks[j]))
    return np.array(dists)


def mapping_aflw2000_to_300w_lp(landmarks):
    landmarks = list(landmarks)
    landmarks = list(reversed(landmarks[:17])) + list(reversed(landmarks[17:27])) + landmarks[27:31] \
                + list(reversed(landmarks[31:36])) + list(reversed(landmarks[36:48])) \
                + list(reversed(landmarks[48:55])) + list(reversed(landmarks[55:60])) \
                + list(reversed(landmarks[60:65])) + list(reversed(landmarks[65:]))
    landmarks = landmarks[:36] + landmarks[38:42] + landmarks[36:38] + landmarks[44:48] + landmarks[42:44] \
                + landmarks[48:]
    return np.array(landmarks)


def mapping_68_points_to_5_points(landmarks):
    new_landmarks = []
    new_landmarks.append((landmarks[44] + landmarks[47]) / 2)
    new_landmarks.append((landmarks[40] + landmarks[37]) / 2)
    new_landmarks.append(landmarks[33])
    new_landmarks.append(landmarks[54])
    new_landmarks.append(landmarks[48])
    return np.array(new_landmarks)


def mapping_68_points_to_15_points(landmarks):
    new_landmarks =        [landmarks[8],
                            landmarks[17], landmarks[21], landmarks[22], landmarks[26],
                            landmarks[30], landmarks[33],
                            landmarks[36], landmarks[39], landmarks[42], landmarks[45],
                            landmarks[60], landmarks[64], landmarks[51], landmarks[57]]
    return np.array(new_landmarks)


def mapping_68_points_to_17_points(landmarks):
    new_landmarks =        [landmarks[0], landmarks[4], landmarks[8], landmarks[12], landmarks[16],
                            landmarks[17], landmarks[21], landmarks[22], landmarks[26],
                            landmarks[27], landmarks[30], landmarks[31], landmarks[35],
                            (landmarks[36]+landmarks[39]) / 2, (landmarks[42]+landmarks[45]) / 2,
                            landmarks[60], landmarks[64]]
    return np.array(new_landmarks)


def mapping_68_points_to_60(landmarks):
    return landmarks[:60]

mapping_points = {
    5: mapping_68_points_to_5_points,
    15: mapping_68_points_to_15_points,
    17: mapping_68_points_to_17_points,
    60: mapping_68_points_to_60,
}



def cosine_3_points(a, b, c):
    # returns cosine between ba and bc
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return cosine_angle
    # angle = np.arccos(cosine_angle)
    # return np.degrees(angle)


def landmarks_to_dist_angles(landmarks):
    x_max, y_max = np.max(landmarks, axis=0)+1
    x_min, y_min = np.min(landmarks, axis=0)-1
    mark_max = np.array((x_max, y_max))
    mark_min = np.array((x_min, y_min))
    diag = euclidean_dist(mark_max, mark_min)
    new_landmarks = []
    for mark in landmarks:
        new_landmarks.append(np.array((euclidean_dist(mark, mark_max) / diag,
                             cosine_3_points(np.array((x_min, y_max)), mark_min, mark))))
    return np.array(new_landmarks)


if __name__ == "__main__":
    p1, p2, p3 = np.array([137.03815386, 382.78470458]), \
                 np.array([137.03815386, 216.37685036]), \
                 np.array([137.03815386, 216.37685036])
    print(cosine_3_points(p1, p2, p3))
