import os.path as path
import sys
from math import fabs

import cv2
import numpy as np

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

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

def align_alignment_boxes(boxes, indices, contours, hierarchy):
    moments_0 = cv2.moments(boxes[0])
    moments_1 = cv2.moments(boxes[1])
    moments_2 = cv2.moments(boxes[2])


capture = cv2.VideoCapture(0)
while True:
    ret,image = capture.read()
    if not ret:
        break
    blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3,3), 1)
    canny =  cv2.Canny(blur, 100, 150, L2gradient=True)

    canny,contours,hier = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for enum,con in enumerate(contours):
        if is_alignment_box(contours, hier, enum):
            cv2.drawContours(image, contours, enum, (255,0,0))

    cv2.imshow('Video', image)
    if cv2.waitKey(10)&0xFF == ord('q'):
        break
cv2.destroyAllWindows()

