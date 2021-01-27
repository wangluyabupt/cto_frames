import os
import numpy as np
import cv2
from ipdb import set_trace
import sys

# 1. get the coors
# 2. compute the offset
# 3. get the src img
# 4. move the src, mix the two pictures into one, save
# info: modify the vid_name and state below

ROOT = sys.argv[1]
# ROOT = "./test1118"
OUT_ROOT = os.path.join(ROOT, 'res')

# bin_path1 = './test1118/bin/IMG-0001-00025.jpg' # bin
# bin_path2 = './test1118/bin/IMG-0001-00026.jpg'
#
# img_path1 = './test1118/src/IMG-0001-00025.jpg' # src
# img_path2 = './test1118/src/IMG-0001-00026.jpg'

bin_dir = os.path.join(ROOT, 'bin')
bin_list= os.listdir(bin_dir)
assert len(bin_list) == 2, 'The num of bin imgs is out of 2!'
bin_ext = bin_list[0].split('.')[-1]
bin_path1 = os.path.join(bin_dir, bin_list[0])
bin_path2 = os.path.join(bin_dir, bin_list[1])

img_dir = os.path.join(ROOT, 'src')
img_list= os.listdir(img_dir)
assert len(img_list) == 2, 'The num of src imgs is out of 2!'
img_ext = img_list[0].split('.')[-1]
img_path1 = os.path.join(img_dir, bin_list[0].split('_')[-1][:-3]+img_ext)
img_path2 = os.path.join(img_dir, bin_list[1].split('_')[-1][:-3]+img_ext)

if not os.path.exists(OUT_ROOT):
    os.makedirs(OUT_ROOT)


bin_img1 = cv2.imread(bin_path1)
bin_img2 = cv2.imread(bin_path2)
if len(bin_img1.shape) > 2:
    bin_img1 = bin_img1[:,:,0]
if len(bin_img2.shape) > 2:
    bin_img2 = bin_img2[:,:,0]

loc1 = np.where(bin_img1>128)
loc2 = np.where(bin_img2>128)
mean1 = np.array([np.mean(loc1[0]), np.mean(loc1[1])]) # h, w
mean2 = np.array([np.mean(loc2[0]), np.mean(loc2[1])])

offset = mean1 - mean2 # h, w

M = np.array([[1, 0, offset[1]],
              [0, 1, offset[0]]], dtype=float)

img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

if len(img1.shape) > 2:
    img1 = img1[:, :, 0]

if len(img2.shape) > 2:
    img2 = img2[:, :, 0]

moved_img2 = cv2.warpAffine(img2, M, img2.shape, borderValue=[255,255,255])
mix_img = cv2.addWeighted(img1, 0.5, moved_img2, 0.5, 0)
id1 = os.path.basename(img_path1)
id2 = os.path.basename(img_path2)
print('id:', id1, id2)
out_img_path = os.path.join(OUT_ROOT, str(id1)+'-'+str(id2)+'.png')
cv2.imwrite(out_img_path, mix_img)
print(out_img_path, 'done!')


