import os
from os import path as path
from random import shuffle
import string

import cv2
import numpy as np

DATA_PATH = path.abspath(path.join(os.getcwd(), 'data/ALPHA/'))
TRAIN_PATH = path.join(DATA_PATH, 'train/lower')
TEST_PATH = path.join(DATA_PATH, 'test/lower')


class LoadData(object):
    '''A class made for importing batches of handwritten letter data.'''

    def __init__(self):
        self.training_images = list()
        for character in string.ascii_lowercase:
            for c in os.listdir(path.join(TRAIN_PATH, character)):
                self.training_images.append((path.abspath(path.join(TRAIN_PATH, '{}/{}'.format(character, c))), string.ascii_lowercase.index(character)))
        shuffle(self.training_images)

    def get_training_batch(self, size):
        ret_arr = list()
        labels = list()

        for a in self.training_images[:min(size, len(self.training_images) - 1)]:
            ret_arr.append(np.resize(np.divide(cv2.imread(a[0], 0), 2.0), (1, 128, 128, 3)))
            labels.append(a[1])

        self.training_images = self.training_images[min(size, len(self.training_images) - 1):]
        return np.concatenate(ret_arr, axis=0), labels
        
    def has_training_data(self):
        return not len(self.training_images) == 0
