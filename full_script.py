# the full script batches image processing in folders, performs analysis, and outputs excel spreadsheet
# image preprocessing and required steps: 
# 1) when capturing images, a object with a known area must included within the image field to convert from pixels to meters (i.e. a clipboard)
# 2) images can be in tiff, png, or jpeg format
# 3) this code requires 2 sets of images: 
#   a) uncropped, original image to calculate pixel-meter scale
#   b) images cropped to flower patch border
# 4) this code uses Lasthenia californica as an example species 
# 5) the full_script final product is an spreadsheet with the following columns: 
#   a) total detected flower count
#   b) flowers/square pixel
#   c) total flower pixels within patch
#   d) total patch pixels
#   e) percent cover = flower pixels/patch pixels
#   f) flowers/ square meter
#   g) area of scale reference (clipboard) in square pixels

import numpy as np # version 1.11.1
import cv2 # version 3.1.0
from skimage.feature import blob_log # version 0.12.3
import matplotlib.pyplot as plt # version 1.5.1
import os 
import skimage.exposure as ex # version 0.12.3
import xlwt 

# batch analyze images in a folder
def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img) # for size and density measurements
            names.append(os.path.join(folder, filename)) # for spreadsheet labeling
    return images, names

# set 1: folder of uncropped images for scaling pixels to meters
images1, names1 = load_images_from_folder('C:/.../Lasthenia_original_tiff')
contour_imgs = images1 
# set 2: folder of cropped images to find size and density of flower patch
images2, names2 = load_images_from_folder('C:/.../Lasthenia_clipped_tiff')
blob_imgs = images2  
paths = names2  

flower_count = []
flowers_pxl_count = []
yellow_pixel_count = []
total_pixel_count = []
cover_count = []
flowers_m2_count = []
clipboard_area_count = []
width_count = []
height_count = []
iter = 0

# percent cover and blob detection
for element in blob_imgs:
    iter = iter + 1
    hsv = cv2.cvtColor(element, cv2.COLOR_BGR2HSV)  # converting to HSV accounts for shadows and lighting
    lower_yellow = np.array([20, 220, 220], np.uint8)  # specific color range for Lasthenia
    upper_yellow = np.array([25, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    final_image = cv2.bitwise_and(element, element, mask=mask)
    yellow_pxl = np.count_nonzero(final_image)
    yellow_pixel_count.append(yellow_pxl)
    total_pixels = np.size(element)
    total_pixel_count.append(total_pixels)
    cover = ((yellow_pxl / total_pixels) * 100) 
    cover_count.append(cover)
    blobs = blob_log(mask, min_sigma=2)  
    total_flowers = 0
    for blob in blobs:
        x, y, r = blob
        if r <= 30:
            total_flowers = (total_flowers + 1)
    flower_count.append(total_flowers)
    flowers_pxl = total_flowers / np.size(element)  # flowers per square pixel in patch
    flowers_pxl_count.append(flowers_pxl)
    fig, ax = plt.subplots(1, 1)
    plt.imshow(element)  # display images to check for errors 
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False) # visualize detected flowers
        ax.add_patch(c)
    print(iter) # display progress 
    
# find contours to scale image
for element in contour_imgs:
    img_copy = element.copy()
    B = element[:, :, 0]  # clipboard isolated in blue channel
    gamma = ex.adjust_gamma(B, gamma=20, gain=1) # gamma adjusted to saturate clipboard 
    kernel = np.ones((5, 5), np.uint8)  # square kernel to maximize alignment with clipboard
    opening = cv2.morphologyEx(gamma, cv2.MORPH_OPEN, kernel=kernel)
    ret, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY) 
    i, cnt, hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # detect clipboard
    areas = [cv2.contourArea(c) for c in cnt] 
    max_index = np.argmax(areas)
    c = cnt[max_index]
    x, y, w, h = cv2.boundingRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width_count.append(w) 
    height_count.append(h)
    clipboard_area = w * h # square pixels found in clipboard
    clipboard_area_count.append(clipboard_area)

for i in range(0, len(flower_count)):
    scale = flowers_pxl_count[i]*(clipboard_area_count[i] / 0.0725705) # 0.0725705 square meters in a standard clipboard
    flowers_m2_count.append(scale)

patch_names = []  # print file name
for path in paths:
    base = os.path.basename(path)  # remove path
    file_wo_ext = os.path.splitext(base)[0]  # remove extension from file name
    patch_names.append(file_wo_ext)

# write excel spreadsheet
book = xlwt.Workbook()
sheet1 = book.add_sheet('Lasthenia')
sheet1.write(0, 0, 'Patch Name')  # column titles
sheet1.write(0, 1, '# of Flowers')
sheet1.write(0, 2, 'Flowers per Square Pixel')
sheet1.write(0, 3, 'Flower Pixels')
sheet1.write(0, 4, 'Patch Pixels')
sheet1.write(0, 5, 'Percent Cover')
sheet1.write(0, 6, 'Flowers per Square Meter')
sheet1.write(0, 7, 'Area of Clipboard (pxls)')
# iterations for each column
A = 0
B = 0
C = 0
D = 0
E = 0
F = 0
G = 0
H = 0
for a in patch_names:
    A = A + 1
    sheet1.write(A, 0, a)
for b in flower_count:
    B = B + 1
    sheet1.write(B, 1, b)
for c in flowers_pxl_count:
    C = C + 1
    sheet1.write(C, 2, c)
for d in yellow_pixel_count:
    D = D + 1
    sheet1.write(D, 3, d)
for e in total_pixel_count:
    E = E + 1
    sheet1.write(E, 4, e)
for f in cover_count:
    F = F + 1
    sheet1.write(F, 5, f)
for g in flowers_m2_count:
    G = G + 1
    sheet1.write(G, 6, g)
for h in clipboard_area_count:
    H = H + 1
    sheet1.write(H, 7, h)
book.save('C:/.../Lasthenia_spreadsheet.xls')

plt.show()
cv2.waitKey(0)
