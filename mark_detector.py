"""Human facial landmark detector based on Convolutional Neural Network."""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, saved_model):
        """Initialization"""
        # A face detector is required for mark detection.
        self.input_size = 128

        # Load the SavedModel object.
        imported = tf.saved_model.load(saved_model)
        self._predict_fn = imported.signatures["serving_default"]
        self._predict_fn._backref_to_saved_model = imported

    def preprocess(self, image):
        """Preprocess the input images."""
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    @tf.function
    def _predict(self, images):
        return self._predict_fn(image_input=images)

    def predict(self, image):
        """Detect marks from images"""

        # Actual detection.
        marks = self._predict(tf.convert_to_tensor([image], dtype=tf.float32))

        # Convert predictions to landmarks.
        marks = np.reshape(marks['dense_1'].numpy(), (-1, 2))

        return marks

    def draw_marks(self, image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)
