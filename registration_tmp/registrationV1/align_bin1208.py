# import os
# import numpy as np
# import cv2
# from ipdb import set_trace
# import sys

# # 1. get the coors
# # 2. compute the offset
# # 3. get the src img
# # 4. move the src, mix the two pictures into one, save
# # info: modify the vid_name and state below

def itera_best2(loc1, loc2, offset, shape, scope=[-3,3], stride=1, smooth=1e-5):
    Iou_Dice = {}
    bin1 = np.empty(shape, dtype=bool)
    bin1.fill(False)
    bin1[loc1] = True
    for i in range(scope[0], scope[1], stride):
        for j in range(scope[0], scope[1], stride):
            # print (i,j)
            off_h = offset[0] + i
            off_w = offset[1] + j
            tmp_loc2 = (loc2[0] + off_h, loc2[1] + off_w)
            tmp_loc2 = valify(tmp_loc2, shape)
            bin2 = np.empty(shape, dtype=bool)
            bin2.fill(False)
            bin2[tmp_loc2] = True
            # compute Iou
            intersection = np.sum(bin1 & bin2) + smooth
            union = np.sum(bin1 | bin2) + smooth
            iou = intersection / union
            Iou_Dice.setdefault((off_h, off_w), {}).update({"iou": iou})

    best_item = sorted(Iou_Dice.items(), key=lambda d: d[1]["iou"])[-1]
    (off_h, off_w) = best_item[0]
    best_iou = best_item[1]['iou']
    return best_iou, off_w, off_h

def valify(loc, shape):
    new_h = []
    new_w = []
    for idx in range(len(loc[0])):
        h = loc[0][idx]
        w = loc[1][idx]
        if 0 <= h < shape[0] and 0 <= w < shape[1]:
            new_h.append(h)
            new_w.append(w)
    return (np.array(new_h,dtype=int), np.array(new_w,dtype=int))



# ROOT = sys.argv[1]
# # ROOT = "./test1118"
# OUT_ROOT = os.path.join(ROOT, 'res')

# # bin_path1 = './test1118/bin/IMG-0001-00025.jpg' # bin
# # bin_path2 = './test1118/bin/IMG-0001-00026.jpg'
# #
# # img_path1 = './test1118/src/IMG-0001-00025.jpg' # src
# # img_path2 = './test1118/src/IMG-0001-00026.jpg'

# bin_dir = os.path.join(ROOT, 'bin')
# bin_list= os.listdir(bin_dir)
# assert len(bin_list) == 2, f'The num of bin imgs is not equal to 2! len(bin_list)={len(bin_list)}'
# bin_ext = bin_list[0].split('.')[-1]
# bin_path1 = os.path.join(bin_dir, bin_list[0])
# bin_path2 = os.path.join(bin_dir, bin_list[1])

# img_dir = os.path.join(ROOT, 'src')
# img_list= os.listdir(img_dir)
# assert len(img_list) == 2, f'The num of src imgs is not equal to 2! len(img_list)={len(img_list)}'
# img_ext = img_list[0].split('.')[-1]
# # img_path1 = os.path.join(img_dir, bin_list[0].split('_')[-1][:-3]+img_ext)
# # img_path2 = os.path.join(img_dir, bin_list[1].split('_')[-1][:-3]+img_ext)
# img_path1 = os.path.join(img_dir, bin_list[0][:-3]+img_ext)
# img_path2 = os.path.join(img_dir, bin_list[1][:-3]+img_ext)

# if not os.path.exists(OUT_ROOT):
#     os.makedirs(OUT_ROOT)

# -*- coding:UTF-8 -*-
import os
import sys
print('sys.path:',sys.path)
import cv2
import numpy as np
# import cv2
from ipdb import set_trace


# 1. get the coors
# 2. compute the offset
# 3. get the src img
# 4. move the src, mix the two pictures into one, save
# info: modify the vid_name and state below

ROOT = sys.argv[1]
# ROOT = "/home/wly/Documents/cto_frames/registration_tmp/registrationV1/test1118/dicoms/dicom14/frames5"

OUT_ROOT = os.path.join(ROOT, 'res')
OUT_MOVED_ROOT = os.path.join(ROOT, 'moved')
# if os.path.exists(OUT_ROOT):
#     for cfile in os.listdir(OUT_ROOT):
#         os.remove(cfile)
# else:
#     os.makedirs(OUT_ROOT)
# bin_path1 = './test1118/bin/IMG-0001-00025.jpg' # bin
# bin_path2 = './test1118/bin/IMG-0001-00026.jpg'F
#
# img_path1 = './test1118/src/IMG-0001-00025.jpg' # src
# img_path2 = './test1118/src/IMG-0001-00026.jpg'

bin_dir = os.path.join(ROOT, 'bin')
bin_list= os.listdir(bin_dir)
print('bin_list:',ROOT)
# assert len(bin_list) == 2, print('The num of bin imgs is not 2!',bin_list)

bin_ext = bin_list[0].split('.')[-1]
bin_path1 = os.path.join(bin_dir, bin_list[0])#bin_list[0]='root_result_dcm_31.png'
bin_path2 = os.path.join(bin_dir, bin_list[1])

# img_dir = os.path.join(ROOT, 'src')
# img_list= os.listdir(img_dir)
img_list=[]
img_dir=ROOT
src_list= os.listdir(img_dir)
for file in src_list:
    if file.endswith('.jpg'):
        img_list.append(file)#img_list[i]='IMG-0003-00001.dcm_5.jpg'

assert len(img_list) == 2, 'The num of src imgs is not 2!'
img_ext = img_list[0].split('.')[-1]
# img_path1 = os.path.join(img_dir, bin_list[0].split('_')[-1][:-3]+img_ext)
# img_path2 = os.path.join(img_dir, bin_list[1].split('_')[-1][:-3]+img_ext)

if img_list[0].split('.')[-2].split('_')[-1] ==bin_list[0].split('.')[-2].split('_')[-1]:
    img_path1 = os.path.join(img_dir,img_list[0])
    img_path2 = os.path.join(img_dir,img_list[1])
else:
    img_path2 = os.path.join(img_dir,img_list[0])
    img_path1 = os.path.join(img_dir,img_list[1])

# if (img_list[0].split('.')[-2].split('_')[-1] == bin_list[0].split('.')[-2].split('_')[-1]):
#     img_path1 = os.path.join(img_dir,img_list[0])
#     img_path2 = os.path.join(img_dir,img_list[1])
# else:
#     img_path2 = os.path.join(img_dir,img_list[0])
#     img_path1 = os.path.join(img_dir,img_list[1])

if not os.path.exists(OUT_ROOT):
    os.makedirs(OUT_ROOT)
    
bin_img1 = cv2.cvtColor(cv2.imread(bin_path1), cv2.cv2.COLOR_BGR2GRAY)
bin_img2 = cv2.cvtColor(cv2.imread(bin_path2), cv2.cv2.COLOR_BGR2GRAY)
# if len(bin_img1.shape) > 2:
#     bin_img1 = bin_img1[:,:,0]
# if len(bin_img2.shape) > 2:
#     bin_img2 = bin_img2[:,:,0]

loc1 = np.where(bin_img1>128)
loc2 = np.where(bin_img2>128)
mean1 = np.array([np.mean(loc1[0]), np.mean(loc1[1])]) # h, w
mean2 = np.array([np.mean(loc2[0]), np.mean(loc2[1])])

offset = mean1 - mean2 # h, w

# coarse iter
iou, off_w, off_h = itera_best2(loc1, loc2, offset, bin_img2.shape, scope=[-25, 25], stride=3)

# fine iter
iou, off_w, off_h = itera_best2(loc1, loc2, [off_h, off_w], bin_img2.shape, scope=[-5, 5], stride=1)

M = np.array([[1, 0, off_w],
              [0, 1, off_h]], dtype=float)


img1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2GRAY)
moved_img2 = cv2.warpAffine(img2, M, img2.shape, borderValue=[255,255,255])
id1 = os.path.basename(img_path1)
id2 = os.path.basename(img_path2)

#add moved 
if not os.path.exists(OUT_MOVED_ROOT):
    os.makedirs(OUT_MOVED_ROOT)
cv2.imwrite(os.path.join(OUT_MOVED_ROOT,id1), img1)
cv2.imwrite(os.path.join(OUT_MOVED_ROOT,id2), moved_img2)

mix_img = cv2.addWeighted(img1, 0.5, moved_img2, 0.5, 0)
print('id:', id1, id2, f'iou:{iou:.4f}', f'offset:{off_h:.2f},{off_w:.2f}')
out_img_path = os.path.join(OUT_ROOT, str(id1)+'-'+str(id2)+'.png')
cv2.imwrite(out_img_path, mix_img)
print(out_img_path, 'done!')






