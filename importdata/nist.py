import os
from os import path as path
from random import shuffle
import string

import cv2
import numpy as np

DATA_PATH = path.abspath(path.join(os.getcwd(), 'data/ALPHA/'))
TRAIN_PATH = path.join(DATA_PATH, 'train')
TEST_PATH = path.join(DATA_PATH, 'test')


training_images = list()

for ch in string.ascii_lowercase:
    for c in os.listdir(path.join(TRAIN_PATH, ch)):
        training_images.append((path.abspath(path.join(TRIAN_PATH, '{}/{}'.format(ch,c))), string.ascii_lowercase.index(ch)))
shuffle(training_images)


def get_training_batch(size):
    ret_arr,labels = [cv2.imread(a),b for a,b in training_images[:min(size, len(training_images)-1)]]
    training_images = training_images[:min(size, len(training_images) - 1)]
    return np.array(ret_arr, dtype=np.uint8), labels

def has_training_data():
    return not len(training_images) == 0
