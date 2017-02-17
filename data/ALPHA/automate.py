'''
A basic python script for moving letter images from their original datastructure
to one that looks like this:
Data
  |Train
  | |lower_{}
  | | |image_name.jpg
  |Test
  | |lower_{}
  | | |image_name.jpg
'''

from math import floor
import os
import os.path as path
import random
import string

RELATIVE_PATHS = ['hsf_0/lower/lower_{}', 'hsf_1/lower/lower_{}', 'hsf_2/lower/lower_{}',
                  'hsf_3/lower/lower_{}', 'hsf_4/lower/lower_{}', 'hsf_6/lower/lower_{}']

ABSOLUTE_PATHS = [path.abspath(path.join('.', rel)) for rel in RELATIVE_PATHS]
NEW_TRAIN = path.abspath(path.join('.', 'fixed/train/lower_{}/'))
NEW_TEST = path.abspath(path.join('.', 'fixed/test/lower_{}/'))

RAND_STR = lambda num: ''.join([random.choice(string.ascii_lowercase) for _ in range(num)])

for nonformat_dir in ABSOLUTE_PATHS:
    for ch in string.ascii_lowercase:
        img_dir = nonformat_dir.format(ch)
        if not path.exists(NEW_TRAIN.format(ch)):
            os.makedirs(NEW_TRAIN.format(ch))
        if not path.exists(NEW_TEST.format(ch)):
            os.makedirs(NEW_TEST.format(ch))
        img_dir_list = os.listdir(img_dir)

        train_num = floor(len(img_dir_list) * 0.8)

        train_images = img_dir_list[:train_num]
        test_images = img_dir_list[train_num:]

        for image_name in train_images:
            img_path = path.join(img_dir, image_name)
            new_path = path.join(NEW_TRAIN.format(ch), RAND_STR(10) + '.png')
            print('Moving {} from {} to {}'.format(image_name, img_dir, new_path))
            os.rename(img_path, new_path)
        for image_name in test_images:
            img_path = path.join(img_dir, image_name)
            new_path = path.join(NEW_TEST.format(ch), RAND_STR(10) + '.png')
            print('Moving {} from {} to {}'.format(image_name, img_dir, new_path))
            os.rename(img_path, new_path)
