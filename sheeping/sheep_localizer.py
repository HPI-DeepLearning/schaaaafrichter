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
        self.font_scale = 0.5
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
        if self.model_type == 'ssd300':
            model = SSD300(n_fg_class=1)
        elif self.model_type == 'ssd512':
            model = SSD512(n_fg_class=1)
        else:
            raise NotImplementedError("Sheep Localizer is not prepared to work with model {}".format(self.model_type))

        model.score_thresh = self._score_threshold

        if self.gpu_id >= 0:
            chainer.backends.cuda.get_device_from_id(self.gpu_id).use()
            model.to_gpu()

        with np.load(self.model_file) as f:
            chainer.serializers.NpzDeserializer(f).load(model)

        self.initialized = True
        self.model = model

    def resize(self, image, is_array=True):
        # image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_CUBIC)
        if is_array:
            image = Image.fromarray(image)
        scale_x = image.size[0] / self.input_size[0]
        scale_y = image.size[1] / self.input_size[1]
        image = image.resize(self.input_size, Image.BICUBIC)
        image = np.asarray(image)
        return image, (scale_x, scale_y)

    def preprocess(self, image, make_copy=True):
        if make_copy:
            image = image.copy()
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        image -= self.mean
        return image

    def localize(self, processed_image):
        if not self.initialized:
            self.build_model()
        bboxes, _, scores = self.model.predict([processed_image])

        return bboxes[0], scores[0]

    def visualize_results(self, image, bboxes, scores, scaling=(1, 1)):
        bboxes = bboxes.astype(np.int32)
        for bbox, score in zip(bboxes, scores):
            if len(bbox) != 4:
                continue

            # scale bounding box with scale factor
            bbox = [bbox[0] * scaling[1], bbox[1] * scaling[0], bbox[2] * scaling[1], bbox[3] * scaling[0]]
            bbox = list(map(lambda x: int(round(x)), bbox))

            width = bbox[3] - bbox[1]
            height = bbox[2] - bbox[0]

            # try to make width of rectangle approximately 0.25% of image size
            thickness_image_size_ratio = 0.0025
            thickness = 1 + round(max(image.shape) * thickness_image_size_ratio)
            cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[1] + width, bbox[0] + height), self.color, thickness)

            score_text = format(float(score), ".2f")
            text_size = cv2.getTextSize(score_text, self.font, self.font_scale, 1)[0]
            text_start = bbox[1] + width - text_size[0], bbox[0] + text_size[1]
            text_end = bbox[1] + width, bbox[0]
            cv2.rectangle(image, text_start, text_end, self.color, -1)
            cv2.putText(image, score_text, text_start, self.font, self.font_scale, (255, 255, 255), bottomLeftOrigin=False)
        return image

