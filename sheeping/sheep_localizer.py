import cv2

import json

import chainer
import numpy as np
from PIL import Image
from chainercv.links.model.ssd.ssd_vgg16 import _imagenet_mean, SSD300, SSD512


class SheepLocalizer:

    def __init__(self, model_file, log_file, gpu_id):
        with open(log_file) as the_log_file:
            self.log = json.load(the_log_file)[0]
        self.model_file = model_file
        self.gpu_id = gpu_id
        self.input_size = tuple(self.log.get('image_size', (512, 512)))
        self.model_type = self.log.get('model_type', 'ssd512')
        self._score_threshold = 0.3
        self.model = None
        self.mean = self.log.get('image_mean', _imagenet_mean)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size_base = 0.5
        self.font_scale = 0.001  # factor for how much the font should scale with image width/height
        self.thickness_base = 1
        self.thickness_scale = 0.0025  # rectangle/font thickness should be approximately 0.25% of image width/height
        self.font_thickness_factor = 0.5  # how much thinner than the rectangle the font should be
        self.color = (0, 255, 0)
        self.initialized = False

    @property
    def score_threshold(self):
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, value):
        self._score_threshold = value
        if self.model is not None:
            self.model.score_thresh = value

    def build_model(self):
        # TODO: determine the correct model type
        model = "correct_model"

        model.score_thresh = self._score_threshold

        # TODO: transfer to GPU if necessary

        # TODO: load weights

        self.initialized = True
        self.model = model

    def resize(self, image, is_array=True):
        if is_array:
            image = Image.fromarray(image)

        # TODO: calculate corresponding scaling factor for x and y (used for scaling back to original size)
        scale_x = 1.0
        scale_y = 1.0

        # TODO: resize to self.input_size (hint: pay attention to resize algorithm)

        image = np.asarray(image)
        return image, (scale_x, scale_y)

    def preprocess(self, image, make_copy=True):
        if make_copy:
            image = image.copy()
        # TODO: reorder channels to the CHW format
        # TODO: convert to the correct dtype (float32)
        # TODO: subtract mean (hint: mean is saved in this class)
        return image

    def localize(self, processed_image):
        if not self.initialized:
            self.build_model()
        # TODO: get bounding boxes and scores
        bboxes = [[0, 0, 100, 100]]
        scores = [[0.9]]

        return bboxes[0], scores[0]

    def visualize_results(self, image, bboxes, scores, scaling=(1, 1)):
        bboxes = bboxes.astype(np.int32)
        for bbox, score in zip(bboxes, scores):
            if len(bbox) != 4:
                continue

            # TODO: scale bounding box with scale factor (see resize function)
            # HINT: the y axis comes first in bounding boxes, order is [top(y), left(x), bottom(y), right(x)]

            # TODO: visualize the found item with a rectangle and render the score as text

        return image

