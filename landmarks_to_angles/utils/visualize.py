import cv2
import math
import numpy as np

def draw_marks(image, marks, color=(255, 0, 0), put_number=True):
    """Draw mark points on image
    https://github.com/yinguobing/head-pose-estimation"""
    for i in range(len(marks)):
        mark = marks[i]
        cv2.circle(image, (int(mark[0]), int(mark[1])), 1, color, -1, cv2.LINE_AA)
        # if i & 1:
        #     continue
        if put_number:
            cv2.putText(image, str(i), (int(mark[0]), int(mark[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                       color , 1, cv2.LINE_AA)


def draw_axes_euler(img, yaw, pitch, roll, tdx=None, tdy=None, size=100,
                    x_color=(0, 0, 255), y_color=(0, 255, 0), z_color=(255, 0, 0)):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if not tdx or not tdy:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right | drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) | drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), x_color, 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), y_color, 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), z_color, 2)
