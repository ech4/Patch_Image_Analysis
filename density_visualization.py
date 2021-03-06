# this code only provides patch density analysis and visualization of individual images
# no excel spreadsheet is generated
# Lasthenia californica is used as an example species 
# see paper: Cruzan, Mitchell B., et al. "Small unmanned aerial vehicles (micro-UAVs, drones) in plant ecology." 
#   Applications in Plant Sciences 4.9 (2016): 1600041.

import skimage.exposure as ex
import skimage.io as io
import numpy as np
import cv2

# input individual images 
img = io.imread('C:/.../patch.tiff')  
gamma = ex.adjust_gamma(img, gamma=10, gain=1) # saturate image to improve color detection 
R = gamma[:, :, 0] # isolate red and blue channels 
B = gamma[:, :, 2]
iso = (R - B) # isolate flower pixels 

# determine percent coverage
red_channel = 0
blue_channel = 0
r_shape = np.shape(R) # total pixels 
b_shape = np.shape(B)
for x in range(1, r_shape[0]):
    for y in range(1, r_shape[1]): 
        if R[x, y] >= 127:
            red_channel = (red_channel + 1)
for x in range(1, b_shape[0]):
    for y in range(1, b_shape[1]):
        if B[x, y] >= 127:
            blue_channel = (blue_channel + 1)
iso_flowers = (red_channel - blue_channel) # total flower pixels 
percent_cover = (iso_flowers / np.size(R)) * 100 # flowers per square pixel
print(percent_cover)

# determine individual flowers in image using contours 
s = np.shape(img)
blank = np.zeros(s)
blank.fill(255) # make blank image for visualization
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(iso, cv2.MORPH_OPEN, kernel)
flag, thresh = cv2.threshold(opening, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # determine flowers
final = cv2.drawContours(blank, contours, -1, (255, 0, 0), 10)
res = cv2.resize(final, (800, 600), interpolation=cv2.INTER_CUBIC)
cv2.imshow('contours', res) # display image to check for errors 
cv2.imwrite('../../Desktop/contours.tiff', blank) # save image 
cv2.waitKey(0)
