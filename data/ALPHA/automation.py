'''
This probably saved me a lot of time, idk tho
'''

import os
from os import path as path

import string

for ch in string.ascii_lowercase:
    dirfrom = path.abspath(path.join(os.getcwd(), 'lower/lower_{}'.format(ch)))
    dirtrain = path.abspath(path.join(os.getcwd(), 'train/lower/{}'.format(ch)))
    dirtest = path.abspath(path.join(os.getcwd(), 'test/lower/{}'.format(ch)))

    images = os.listdir(dirfrom)
    num_train = int(len(images) * 0.80)

    train_images = images[:num_train]
    test_images = images[num_train:]

    for img in train_images:
        newpath = path.join(dirtrain, img)
        oldpath = path.join(dirfrom, img)
        os.rename(oldpath, newpath)

    for img in test_images:
        newpath = path.join(dirtest, img)
        oldpath = path.join(dirfrom, img)
        os.rename(oldpath, newpath)
