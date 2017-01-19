'''
Created on Wednesday, January 4, 2017 at 22:14.
Author: Aramis Tanelus
'''

import math
import os
import statistics
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from . import kmeans


def find_digits(image, display=False):
    '''
    Takes an image and returns a list of images containing contrasting marks found.
    This assumes the paper/background is white
    Arguments:
    image: an ndarray representing the image to examine
    display: an optional bool that controls whether the found digits will be plotted and displayed
    '''

    #Create a copy of the image
    img = np.copy(image)

    #Create two variables for convenience
    sh_0 = img.shape[0]
    sh_1 = img.shape[1]

    #Ensure the image is grayscale
    if len(image.shape) == 3:
        #The image is in BGR format
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Apply a blur
    #img = cv2.GaussianBlur(img, ((sh_0 // 175) | 1, (sh_1 // 175) | 1), sigmaX=4)
    img = cv2.medianBlur(img, ((sh_0 * sh_1)//1306667) | 1)

    #Apply an adaptive threshold with the inverse binary rule
    # max(...,...) | 1 is to ensure the block size is odd, this assumes the number is positive
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, max((sh_0 * sh_1) // 34560, 3) | 1, (sh_0 * sh_1) //518400)

    #Find all contours in the image, this modifies the threshold image so a copy will be used
    _, contours, _ = cv2.findContours(np.copy(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c)  for c in contours]
    k_means = kmeans.kmeans_1d(areas, 2)
    stdev_areas = statistics.stdev(areas)

    '''
    ##DEBUG START##
    print(k_means)
    ##DEBUG END##
    '''

    #Create a list of the contours whos areas are within one standard deviation of the mean
    #letter_contours = [c for c in contours if math.fabs(cv2.contourArea(c) - max(k_means.keys())) < 0.5 * stdev_areas]
    letter_contours = [c for c in contours if cv2.contourArea(c) > min(k_means.keys()) + stdev_areas * 0.25]

    print(len(letter_contours))

    #Create a list to hold the images that will be returned
    letter_images = list()

    #Loop through letter_contours and generate a blank image containing only the contour
    for letter in letter_contours:
        blank = np.zeros_like(img)
        cv2.drawContours(blank, [letter], 0, 255, 20)
        x, y, w, h = cv2.boundingRect(letter)
        box_size = max(w,h)

        #Important note, numpy uses rows as x and columns as y, opposite of opencv
        letter_box = blank[y : min(sh_0, y + box_size), x : min(sh_1, x + box_size)].copy()
        
        '''
        ##DEBUG START##
        print(letter_box.shape)
        print('sh0-{} sh1-{}'.format(sh_0,sh_1))
        print('X{} Y{} W{} H{}'.format(x,y,w,h))
        ##DEBUG END##
        '''

        #Resize the image to be 28x28 pixels if it isn't
        if box_size != 28:
            letter_box = cv2.resize(letter_box, (28,28), interpolation=cv2.INTER_CUBIC)
        
        #Moments of the contour
        M = cv2.moments(cv2.findContours(letter_box, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1][0])

        #Now proceed to translate the contour so that the center of mass is in the center of the image (14,14)
        cm_x = round(M['m10']/M['m00'])
        image_translated = __translate_image__(letter_box, 14 - cm_x, 0)
        letter_images.append(image_translated)

    #Display the images with matplotlib if display is True
    if display:
        _, plots = plt.subplots(math.ceil(len(letter_images)/2), 2)
        for count, subplot in enumerate(np.resize(plots, plots.shape[0] * plots.shape[1])):
            if count >= len(letter_images):
                break
            subplot.imshow(letter_images[count], cmap='gray')
            subplot.axes.get_xaxis().set_ticks([])
            subplot.axes.get_yaxis().set_ticks([])
        plt.show()
    
    return letter_images


def write_digits(images, num):
    '''
    Arguments:
    image: The ndarray representing the image to write to
    contour_bounding_rect: the tuple representing the contour's bounding box
    num: an int representing the digit predicted to be represented by the contour 
    '''
    
    _, plots = plt.subplots(math.ceil(len(images)/2), 2)
    for count, subplot in enumerate(np.resize(plots, plots.shape[0] * plots.shape[1])):
        if count >= len(num):
            break
        subplot.set_title(str(num[count]))
        subplot.axes.get_xaxis().set_ticks([])
        subplot.axes.get_yaxis().set_ticks([])
        subplot.imshow(images[count], cmap='gray')
    plt.show()

def img_from_path(path):
    '''
    Arguments:
    path: a string representing the absolute path to the image
    '''
    return cv2.imread(path)

def __translate_image__(image, a: int, b: int):
    '''
    Transforms an image using the rule (x,y) -> (x + a, y + b)
    Arguments:
    image: an ndarray representing the image to be translated
    a: an int that determines how much to translate the image on the x axis
    b: an int that determines how much to translate the image on the y axis
    '''
    S_0,S_1 = image.shape
    return cv2.warpAffine(image, np.array([[1,0,a],[0,1,b]], dtype=np.float32), (S_1,S_0))

def to_numpy(image):
    '''
    Flips an image (x,y) -> (y,x) and divides it by 255 because the inconsistency between cv and numpy is fucking annoying
    Arguments:
    image: an ndarray representing the image to be modified
    '''
    #return np.copy(np.swapaxes(image, 0, 1).astype(np.float32) / 255)
    return image.astype(np.float32) / 255
