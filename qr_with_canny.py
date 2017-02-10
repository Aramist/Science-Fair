import os.path as path
import sys
from math import fabs

import cv2
import numpy as np

IMAGE_PATH = path.abspath(path.join('.', 'qr_code_cropped.jpg'))
WINDOWS = ['Contours']

for window_name in WINDOWS:
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

image = cv2.imread(IMAGE_PATH)
blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3,3), 1)
canny =  cv2.Canny(blur, 100, 150, L2gradient=True)

'''
cv2.imshow('Original', image)
cv2.imshow('Blur', blur)
cv2.imshow('Canny', canny)
'''

canny,contours,hier = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, contours, -1, (255,0,0))
#cv2.imshow('Original', image)
'''
for enum,con in enumerate(contours):
    copy = image.copy()
    cv2.drawContours(copy, [con], 0, (255,0,0))
    cv2.putText(copy, '{}: {}'.format(enum, hier[0,enum]), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    cv2.imshow('Contours', copy)
    if cv2.waitKey(0) == ord('q'):
        break
'''

def about_equal(one, two, tolerance=.10):
    return fabs(one-two) < tolerance * two

def is_alignment_box(contours, hierarchy, index):
    sub = hier[0][enum]
    child_present = sub[2] != 1
    child = contours[sub[2]]
    
    sub1 = hier[0][sub[2]]
    only_child = sub1[0] == sub1[1] == -1
    child1_present = sub1[2] != -1
    
    sub2 = hier[0][sub1[2]]
    only_child1 = sub2[0] == sub2[1] == -1
    child2_present = sub2[2] != -1

    sub3 = hier[0][sub2[2]]
    only_child2 = sub3[0] == sub3[1] == -1
    child3_present = sub3[2] != -1

    sub4 = hier[0][sub3[2]]
    only_child3 = sub4[0] == sub4[1] == -1
    child4_present = sub4[2] != -1

    sub5 = hier[0][sub4[2]]
    only_child4 = sub5[0] == sub5[1] == -1

    flags = child_present,only_child,child1_present,only_child1 \
    ,child2_present,only_child2,child3_present,only_child3 \
    ,child4_present,only_child4
    if not all(flags):
        return False

    print('49/25: {}\n25/9: {}'.format(49/25,25/9))
    area_outer = cv2.contourArea(contours[index])
    area_inner = cv2.contourArea(contours[sub2[2]])
    area_ratio = area_outer/area_inner
    error = fabs((area_ratio - 49/25) * 25/49)
    outer_ratio_complies = error < 0.20
    
    area_inner2 = cv2.contourArea(contours[sub4[2]])
    area_ratio_2 = area_inner/area_inner2
    error1 = fabs((area_ratio_2 - 25/9) * 9/25)
    inner_ratio_complies = error1 < 0.20
    
    flags = inner_ratio_complies, outer_ratio_complies
    return all(flags)


def determine_alignment(boxes, indices):
    mo_0 = cv2.moments(boxes[0])
    mo_1 = cv2.moments(boxes[1])
    mo_2 = cv2.moments(boxes[2])

    cm_0 = (mo_0['10']/mo_0['00'], mo_0['01']/mo_0['00'])
    cm_1 = (mo_1['10']/mo_1['00'], mo_1['01']/mo_1['00'])
    cm_2 = (mo_2['10']/mo_2['00'], mo_2['01']/mo_2['00'])

    slope_01 = (cm_0[1] - cm_1[1])/(cm_0[0] - cm_1[0])
    slope_12 = (cm_1[1] - cm_2[1])/(cm_1[0] - cm_2[0])
    slope_20 = (cm_2[1] - cm_0[1])/(cm_2[0] - cm_0[0])

    #Normality testing
    perp_0112 = about_equal(-1/slope_01, slope_12)
    perp_1220 = about_equal(-1/slope_12, slope_20)
    perp_2001 = about_equal(-1/slope_20, slope_01)

    #Qr Code box numbering:
    # |1||2|
    # ||||||
    # |0||||

    if perp_0112:
        qr_1 = boxes[1]
        if slope_12 > 0:
            qr_2 = boxes[2]
            qr_0 = boxes[0]
        else:
            qr_2 = boxes[0]
            qr_0 = boxes[2]
    if perp_1220:
        qr_1 = boxes[2]
        if slope_20 > 0:
            qr_2 = boxes[0]
            qr_0 = boxes[1]
        else:
            qr_2 = boxes[1]
            qr_0 = boxes[0]
    if perp_2001:
        qr_1 = boxes[0]
        if slope_01 > 0:
            qr_2 = boxes[1]
            qr_0 = boxes[2]
        else:
            qr_2 = boxes[2]
            qr_0 = boxes[1]
    if qr_0 is None or qr_1 is None or qr_2 is None:
        return False
    return (qr_0, qr_1, qr_2)

        
    

for enum,con in enumerate(contours):
    if is_alignment_box(contours, hier, enum):
        boxes_ordered = determine_alignment
        cv2.drawContours(image, contours, enum, (255,0,0))

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

