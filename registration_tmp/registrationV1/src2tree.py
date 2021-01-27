# # -*- coding: utf-8 -*-
#
# import os
# import numpy as np
# import sys
#
# # print ('step into segmentation...')
# # SRC_ROOT = '/home/DataBase3/ww/0818/data/ZHU_BIN'
# POS = 'RIGHT'
# # cmd = f"python /home/DataBase3/ww/0818/BASNet-master/basnet_test.py --P {POS} --I_R {SRC_ROOT}"
# # os.system(cmd)
# # print ('segmentation done!')
# # print (os.getcwd())
#
# # bin_dir  = os.path.join(SRC_ROOT,POS,'binary')+'/'
# # RES_ROOT = os.path.join(SRC_ROOT,POS,'result')
# # bin_dir  = r'/home/ww/FindRoot/dataset/binary'
#
# SRC_ROOT = sys.argv[1]
# # SRC_ROOT = r'./test1118'
# RES_ROOT = SRC_ROOT
# if not os.path.exists(RES_ROOT):
#     os.makedirs(RES_ROOT)
#
# # src_path = os.path.join(SRC_ROOT,POS)
# src_path_ROOT = os.path.join(SRC_ROOT, 'src_3')
# src_list = os.listdir(src_path_ROOT)
# res_dir = RES_ROOT
# res_file_path = os.path.join(res_dir, 'root.txt')
# RES_LocateRoot = os.path.join(res_dir, 'bin')
# if not os.path.exists(RES_LocateRoot):
#     os.makedirs(RES_LocateRoot)
# print('locate root...')
# # cmd = f"python ./extractCenterline/locate_root.py --I_R {src_path_ROOT} \
# #         --rf_path {res_file_path} --O_R {RES_LocateRoot}"
# # os.system(cmd)
#
# os.system('python ./extractCenterline/locate_root.py --I_R {} --rf_path {} --O_R {}'.format(src_path_ROOT,res_file_path,RES_LocateRoot))
#
#
# print('done!')
# # pre_deal.deal(res_dir) # 预处理
# # print (os.getcwd())
#
# # bin_dir_ROOT = r'/home/ww/FindRoot/dataset/binary'
# # bin_list = os.listdir(bin_dir_ROOT)
# # for dir in bin_list:
# #     bin_dir = os.path.join(bin_dir_ROOT,dir)
# #     res_dir = os.path.join(RES_ROOT, dir)
# #     if not os.path.exists(res_dir):
# #         os.makedirs(res_dir)
# #     res_file_path = os.path.join(res_dir, 'root.txt')
# #     RES_buildtree = os.path.join(res_dir, 'tree')
# #     if not os.path.exists(RES_buildtree):
# #         os.makedirs(RES_buildtree)
# #     print ('build tree...')
# #     cmd  = f"python /home/DataBase3/ww/0818/extractCenterline/bin2line.py --P {POS} \
# #             --I_R {bin_dir} --rf_path {res_file_path} --O_R {RES_buildtree}"
# #     os.system(cmd)
# #     print ('done!')
# # print (os.getcwd())
# #
# # deal()


# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
# print ('step into segmentation...')
# SRC_ROOT = '/home/DataBase3/ww/0818/data/ZHU_BIN'
POS = 'RIGHT'
# cmd = f"python /home/DataBase3/ww/0818/BASNet-master/basnet_test.py --P {POS} --I_R {SRC_ROOT}"
# os.system(cmd)
# print ('segmentation done!')
# print (os.getcwd())

# bin_dir  = os.path.join(SRC_ROOT,POS,'binary')+'/'
# RES_ROOT = os.path.join(SRC_ROOT,POS,'result')
# bin_dir  = r'/home/ww/FindRoot/dataset/binary'

SRC_ROOT = sys.argv[1]
# SRC_ROOT = r'./test1118'
# SRC_ROOT = r'./test1118/dicoms/dicom1/framesi'
RES_ROOT = SRC_ROOT
if not os.path.exists(RES_ROOT):
    os.makedirs(RES_ROOT)

# src_path = os.path.join(SRC_ROOT,POS)
src_path_ROOT = SRC_ROOT
# src_list = os.listdir(src_path_ROOT)
res_dir = RES_ROOT
res_file_path = os.path.join(res_dir, 'root.txt')
RES_LocateRoot = os.path.join(res_dir, 'bin')
if not os.path.exists(RES_LocateRoot):
    os.makedirs(RES_LocateRoot)

print('locate root...')
# cmd = f"python ./extractCenterline/locate_root.py --I_R {src_path_ROOT} \
#         --rf_path {res_file_path} --O_R {RES_LocateRoot}"
# os.system(cmd)

# os.system('python ./extractCenterline/locate_root.py --I_R {} --rf_path {} --O_R {}'.format(src_path_ROOT,res_file_path,RES_LocateRoot))

import os
# os.chdir('/home/DataBase3/ww/0818/extractCenterline')
import numpy as np
from PIL import Image
from matplotlib import pyplot
# from lib.findroot import *
# from ipdb import set_trace
import cv2
import glob
import argparse

from registration_tmp.registrationV1.extractCenterline.lib.findroot import BinarySegment, FindRoot


def get_root (image,img_name,findroot_cls):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_result = seg_cls.inference(image)
    return findroot_cls.get_root(image_result,img_name)

IMG_ROOT = src_path_ROOT
FILE_PATH = res_file_path
ROOT_OUT_ROOT = RES_LocateRoot

if not os.path.exists(ROOT_OUT_ROOT):
    os.makedirs(ROOT_OUT_ROOT)
if os.path.exists(FILE_PATH):
    os.remove(FILE_PATH)

seg_cls = BinarySegment()
findroot_cls = FindRoot(OUT_ROOT=ROOT_OUT_ROOT)
with open(FILE_PATH, 'w') as root_f:
    # img_path_list = glob.glob(IMG_ROOT + '/' + '*.png')
    img_path_list = os.listdir(IMG_ROOT)
    img_path_list = list(filter(lambda x: ('.jpg' in x) or ('.png' in x),img_path_list))
    img_path_list = [os.path.join(IMG_ROOT,x) for x in img_path_list]
    for img_path in img_path_list:
        print(img_path)
        img = np.array(Image.open(img_path))
        img_name = os.path.basename(img_path).split('.')[0]
        if len(img.shape) > 2 and img.shape[-1] == 3:
            png_img = img[:, :, 0]
        else:
            png_img = img
        root = get_root(png_img, img_name + '.png',findroot_cls)
        root_f.writelines(img_name + ' ' + str(root) + '\n')
        # print (img_name,root)


print('done!')
# pre_deal.deal(res_dir) # 预处理
# print (os.getcwd())

# bin_dir_ROOT = r'/home/ww/FindRoot/dataset/binary'
# bin_list = os.listdir(bin_dir_ROOT)
# for dir in bin_list:
#     bin_dir = os.path.join(bin_dir_ROOT,dir)
#     res_dir = os.path.join(RES_ROOT, dir)
#     if not os.path.exists(res_dir):
#         os.makedirs(res_dir)
#     res_file_path = os.path.join(res_dir, 'root.txt')
#     RES_buildtree = os.path.join(res_dir, 'tree')
#     if not os.path.exists(RES_buildtree):
#         os.makedirs(RES_buildtree)
#     print ('build tree...')
#     cmd  = f"python /home/DataBase3/ww/0818/extractCenterline/bin2line.py --P {POS} \
#             --I_R {bin_dir} --rf_path {res_file_path} --O_R {RES_buildtree}"
#     os.system(cmd)
#     print ('done!')
# print (os.getcwd())
#
# deal()
