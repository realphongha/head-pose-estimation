"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
import cv2
import numpy as np
import time
import os
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from pathlib import Path

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

print("OpenCV version: {}".format(cv2.__version__))

CNN_INPUT_SIZE = 128

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--img", type=str, default=None,
                    help="Image file to be processed.")
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument('--show', action='store_true', default=False,
                    help="Show result or not?")
parser.add_argument('--store', action='store_true', default=False,
                    help="Store result or not?")
parser.add_argument('--speed', action='store_true', default=False,
                    help="Display speed or not?")
args = parser.parse_args()

# Introduce mark_detector to detect landmarks.
mark_detector = MarkDetector()


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def predict_img():
    img = cv2.imread(args.img)

    height, width = img.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    facebox = mark_detector.extract_cnn_facebox(img)

    if facebox is not None:
        # Detect landmarks from image of 128x128.
        face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
        # cv2.imshow("Preview", face_img)
        # cv2.waitKey(2000)
        face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        marks = mark_detector.detect_marks(face_img)

        # Convert the marks locations from local CNN to global image.
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]

        # Uncomment following line to show raw marks.
        # mark_detector.draw_marks(img, marks, color=(0, 255, 0))

        # Uncomment following line to show facebox.
        # mark_detector.draw_box(img, [facebox])

        # Try pose estimation with 68 points.
        pose = pose_estimator.solve_pose_by_68_points(marks)

        # Uncomment following line to draw pose annotation on img.
        pose_estimator.draw_annotation_box(
            img, pose[0], pose[1], color=(255, 128, 128))

        # Uncomment following line to draw head axes on img.
        # pose_estimator.draw_axes(img, steady_pose[0], steady_pose[1])

        if args.show:
            cv2.imshow("Preview", img)
            cv2.waitKey(5000)
        if args.store:
            filename = Path(args.img).stem
            cv2.imwrite(os.path.join("output", "%s-%d.jpg" % (filename, int(time.time()))), img)


def predict_video():
    """MAIN"""
    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # prepare the video writer:
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if args.store and args.video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        filename = Path(args.video).stem
        out = cv2.VideoWriter(os.path.join("output", "%s-%d.avi" % (filename, int(time.time()))),
                                           fourcc, fps, (width, height))

    _, sample_frame = cap.read()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    # count how many frame predicted:
    i = 0
    # start time:
    start = time.perf_counter()
    while True:
        i += 1
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()

        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

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
                frame, steady_pose[0], steady_pose[1], color=(128, 255, 128)
            )

            # Uncomment following line to draw head axes on frame.
            # pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])

        # Show preview.
        if args.show:
            cv2.imshow("Preview", frame)
            if cv2.waitKey(10) == 27:
                break

        if args.store and args.video:
            out.write(frame)
        if args.speed:
            if i % fps == fps - 1:
                print("Predicted %d frame(s) in %.5f second(s). %.5f frames/s" %
                      (i, time.perf_counter() - start, i / (time.perf_counter() - start)))

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()

    # Release the writer:
    if args.store and args.video:
        out.release()


def main():
    if args.img:
        if args.speed:
            start = time.perf_counter()
            for i in range(100):
                predict_img()
                print("Processed %d image(s)" % (i + 1))
            print("Avg: %.5f image(s) per second" % (1000 / (time.perf_counter() - start)))
        else:
            predict_img()
    elif args.video or args.cam:
        # multiprocessing may not work on Windows and macOS, check OS for safety.
        detect_os()
        predict_video()


if __name__ == '__main__':
    main()
