"""Object detector based on EfficientDet model.

This module supports inferenceing with the official EfficientDet model. 
For more details: https://github.com/yinguobing/efficientdet-runner
"""
import cv2
import numpy as np
import tensorflow as tf


class FaceDetector(object):
    """Mini module to run the EfficientDet model."""

    def __init__(self, saved_model):
        """Build an EfficientDet model runner.

        Args:
            saved_model: the string path to the SavedModel.
        """
        self.scale_width = 0
        self.scale_height = 0
        self.input_size = 512

        # Load the SavedModel object.
        imported = tf.saved_model.load(saved_model)
        self._predict_fn = imported.signatures["serving_default"]

        # To avoid garbage collected by Python, see TensorFlow issue:37615
        self._predict_fn._backref_to_saved_model = imported

    def preprocess(self, image):
        """Preprocess the input image."""

        # Scale the image first.
        height, width, _ = image.shape
        self.ratio = self.input_size / max(height, width)
        image = cv2.resize(
            image, (int(self.ratio * width), int(self.ratio * height)))

        # Convert to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Then pad the image to input size.
        self.padding_h = self.input_size - int(self.ratio * width)
        self.padding_v = self.input_size - int(self.ratio * height)
        image = cv2.copyMakeBorder(
            image, 0, self.padding_v, 0, self.padding_h, cv2.BORDER_CONSTANT, (0, 0, 0))

        return image

    def __filter(self, detections, threshold):
        """Filter the detection results by score threshold."""
        # Get the detection results.
        boxes = detections['output_0'].numpy()[0]
        scores = detections['output_1'].numpy()[0]
        classes = detections['output_2'].numpy()[0]

        # Filter out the results by score threshold.
        mask = scores > threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        return boxes, scores, classes

    @tf.function
    def _predict(self, images):
        return self._predict_fn(images)

    def predict(self, image, threshold):
        """Run inference with image inputs.

        Args:
            image: a numpy array as an input image.

        Returns:
            predictions: result.
        """
        frame_tensor = tf.constant(image, dtype=tf.uint8)
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)
        detections = self._predict(frame_tensor)
        boxes, scores, classes = self.__filter(detections, threshold)

        # Scale the box back to the original image size.
        boxes /= self.ratio

        # Crop out the padding area.
        boxes[:, 2] = np.minimum(
            boxes[:, 2], (self.input_size - self.padding_v)/self.ratio)
        boxes[:, 1] = np.minimum(
            boxes[:, 1], (self.input_size - self.padding_h)/self.ratio)

        return boxes, scores, classes

    def transform_to_square(self, boxes, scale=1.0, offset=(0, 0)):
        """Get the square bounding boxes.

        Args:
            boxes: input boxes [[ymin, xmin, ymax, xmax], ...]
            scale: ratio to scale the boxes
            offset: a tuple of offset to move the boxes (x, y)

        Returns:
            square boxes.
        """
        ymins, xmins, ymaxs, xmaxs = np.split(boxes, 4, 1)
        width = xmaxs - xmins
        height = ymaxs - ymins

        # How much to move.
        offset_x = offset[0] * width
        offset_y = offset[1] * height

        # Where is the center location.
        center_x = np.floor_divide(xmins + xmaxs, 2) + offset_x
        center_y = np.floor_divide(ymins + ymaxs, 2) + offset_y

        # Make them squares.
        margin = np.floor_divide(np.maximum(height, width) * scale, 2)
        boxes = np.concatenate((center_y-margin, center_x-margin,
                                center_y+margin, center_x+margin), axis=1)

        return boxes

    def clip_boxes(self, boxes, margins):
        """Clip the boxes to the safe margins.

        Args:
            boxes: input boxes [[ymin, xmin, ymax, xmax], ...].
            margins: a tuple of 4 int (top, left, bottom, right) as safe margins.

        Returns:
            boxes: clipped boxes.
            clip_mark: the mark of clipped sides.
        """
        top, left, bottom, right = margins

        clip_mark = (boxes[:, 0] < top, boxes[:, 1] < left,
                     boxes[:, 2] > bottom, boxes[:, 3] > right)

        boxes[:, 0] = np.maximum(boxes[:, 0], top)
        boxes[:, 1] = np.maximum(boxes[:, 1], left)
        boxes[:, 2] = np.minimum(boxes[:, 2], bottom)
        boxes[:, 3] = np.minimum(boxes[:, 3], right)

        return boxes, clip_mark

    def draw_box(self, image, box, score, color=(0, 255, 0)):
        """Draw the bounding box.

        Args:
            boxes: the face box.
            color: the color of the box.
            scores: detection score.

        """
        y0, x0, y1, x1 = [int(b) for b in box]
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(image, "Face:{:.2f}".format(score),
                    (x0, y0-7), cv2.FONT_HERSHEY_DUPLEX, 0.5, color,
                    1, cv2.LINE_AA)
