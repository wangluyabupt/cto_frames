'''
This code is to transfer the segmentation results into 0 or 255 pixel values with single channel.
'''

import os
import numpy as np
import cv2
from ipdb import set_trace

IMG_ROOT = './dataset/pix2pix/CRA'

img_name_list = os.listdir(IMG_ROOT)
for img_name in img_name_list:
    img_path = os.path.join(IMG_ROOT,img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    tmp_loc = np.where(img>128)
    new_img = np.zeros_like(img)
    new_img[tmp_loc] = 255
    cv2.imwrite(img_path,new_img)
    print(img_name,'done!')

