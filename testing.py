import sys
import os

import numpy as np
import cv2

from computervision import detection
from deeplearning import prediction
from deeplearning.prediction import abs_path

IMAGES = ['handwriting_2.jpg', 'handwriting_3.jpg']
IMAGE_PATHS = [abs_path('testing images/' + _x_) for _x_ in IMAGES]

img = cv2.imread(IMAGE_PATHS[0])
letters = detection.find_digits(img, display=True)
#labels = [prediction.test_image(detection.to_numpy(im)) for im in letters]
labels = [prediction.test_image(letters[0])]
detection.write_digits(letters, labels)