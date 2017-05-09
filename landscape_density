# this is an simplification of the full script to determine landscape flower density from an aerial image (i.e. UAV, satellite) 
# there are two species examples provided: Lasthenia californica (yellow flowers) and Plagiobothrys (white flowers) 

import numpy as np
import cv2
from skimage.feature import blob_log
import matplotlib.pyplot as plt
import os
import skimage.exposure as ex

img = cv2.imread('C:/.../aerial_image.tif')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # reduces variation in light exposure in the image

lower = np.array([20, 220, 220], np.uint8) # color range for Lasthenia
upper = np.array([25, 255, 255], np.uint8)

# lower = np.array([10,165, 185], np.uint8) # color range for Plagiobothrys 
# upper = np.array([35, 250, 255], np.uint8)

mask = cv2.inRange(hsv, lower, upper)
final = cv2.bitwise_and(img, img, mask=mask)
res = cv2.resize(final, (1175, 750), cv2.INTER_CUBIC) 
res2 = cv2.resize(img, (1175, 750), cv2.INTER_CUBIC) 
cv2.imwrite("C:/.../flower_pixels.tif", mask) # save image
cv2.imshow('yellow', res) # image displaying pixels within denoted range
cv2.imshow('original', res2) # original image 
cv2.waitKey(0)
