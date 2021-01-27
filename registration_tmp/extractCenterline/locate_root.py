import os
# os.chdir('/home/DataBase3/ww/0818/extractCenterline')
import numpy as np
from matplotlib import pyplot
from lib.findroot import *
# from ipdb import set_trace
import cv2
import glob
import argparse

def get_root (image,img_name,findroot_cls):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_result = seg_cls.inference(image)
    return findroot_cls.get_root(image_result,img_name)

parse = argparse.ArgumentParser()
parse.add_argument("--I_R", type=str, default=None)
parse.add_argument("--rf_path", type=str, default=None)
parse.add_argument("--O_R", type=str, default=None)

args = parse.parse_args()
IMG_ROOT = args.I_R
FILE_PATH = args.rf_path
ROOT_OUT_ROOT = args.O_R


# IMG_ROOT = '/mnt/dataBase3/wrj/segmentation/Data/0520data/0916res_jpg/CAU' # src img
# ROOT_OUT_ROOT = '/mnt/dataBase3/wrj/segmentation/Data/0520data/0916res_jpg/result/CAU/locate_root'
# FILE_PATH = '/mnt/dataBase3/wrj/segmentation/Data/0520data/0916res_jpg/result/CAU/root.txt'

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



