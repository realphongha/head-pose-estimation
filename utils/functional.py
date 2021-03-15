import cv2
import numpy as np
import math


def r_vec_to_euler(r_vec):
    # converts rotation vector to euler angles
    r_mat, _ = cv2.Rodrigues(r_vec)
    p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
    _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
    pitch, yaw, roll = u_angle.flatten()

    # I do not know why the roll axis seems flipped 180 degree. Manually by pass this issue.
    if roll > 0:
        roll = 180 - roll
    elif roll < 0:
        roll = -(180 + roll)
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return pitch, yaw, roll


if __name__ == "__main__":
    pass