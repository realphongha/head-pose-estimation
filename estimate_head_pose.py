"""Demo code shows how to estimate human head pose.

Three major steps for this code.

Step 1: face detection. The human faces are detected by a deep learning face
    face_detector .Then the face boxes are modified a little to suits the need of
    landmark detection.
Step 2: facial landmark detection. This is done by a custom Convolutional
    Neural Network trained with TensorFlow.
Step 3: head pose estimation. The pose is estimated by solving a PnP problem.

All models and training code are available at: https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

from face_detector import FaceDetector
from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

print("OpenCV version: {}".format(cv2.__version__))

devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

# Parse arguments from user inputs.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


def main():
    """Run human head pose estimation from video files."""

    # What is the threshold value for face detection.
    threshold = 0.5

    # Setup the video source. If no video file provided, the default webcam will
    # be used.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # If reading frames from a webcam, try setting the camera resolution.
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Get the real frame resolution.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Introduce a mark_face_detector to detect face marks.
    detector_mark = MarkDetector("assets/mark_model")

    # Introduce a face face_detector to detect human faces.
    detector_face = FaceDetector("assets/face_model")

    # Introduce pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # Introduce a metter to measure the FPS.
    tm = cv2.TickMeter()

    # Loop through the video frames.
    while True:
        tm.start()

        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Preprocess the input image.
        _image = detector_face.preprocess(frame)

        # Run the model
        boxes, scores, classes = detector_face.predict(_image, threshold)

        # Transform the boxes into squares.
        boxes = detector_face.transform_to_square(
            boxes, scale=1.22, offset=(0, 0.13))

        # Clip the boxes if they cross the image boundaries.
        boxes, _ = detector_face.clip_boxes(boxes, (0, 0, height, width))

        # Detect facial marks.
        if boxes.size > 0:
            # Get one face image.
            facebox = boxes[0]
            top, left, bottom, right = [int(x) for x in facebox]
            face_image = frame[top:bottom, left:right]

            # Run detection.
            face_image = detector_mark.preprocess(face_image)
            marks = detector_mark.predict(face_image)

            # Convert the marks locations from local CNN to global image.
            marks *= (right - left)
            marks[:, 0] += left
            marks[:, 1] += top

            # Uncomment following line to show facebox.
            # detector_face.draw_box(frame, facebox, scores[0])

            # Uncomment following line to show raw marks.
            detector_mark.draw_marks(frame, marks, color=(0, 255, 0))

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Uncomment following line to draw pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            pose_estimator.draw_annotation_box(
            frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # Uncomment following line to draw head axes on frame.
            pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])

        tm.stop()

        # Draw FPS on the screen's top left corner.
        cv2.putText(frame, "FPS: {:.0f}".format(tm.getFPS()), (24, 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()
