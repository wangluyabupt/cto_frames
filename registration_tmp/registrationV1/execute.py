import os
import os
# os.chdir('/home/DataBase3/ww/0818/extractCenterline')
import numpy as np
from PIL import Image
from matplotlib import pyplot
from tqdm import tqdm
# from lib.findroot import *
# from ipdb import set_trace
import cv2
import glob
import argparse
import shutil
os.system(
    'export PYTHONPATH=$PYTHONPATH:/home/wly/Documents/cto_frames/registration_tmp'
)
os.system(
    'export PYTHONPATH=$PYTHONPATH:/home/wly/Documents/cto_frames/registration_tmp/extractCenterline'
)
os.system('export PATH=/home/wly/anaconda3/bin:$PATH')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# print(os.path.abspath(os.curdir))
# print(os.getcwd())
# print(os.path.abspath(__file__))

os.chdir("/home/wly/Documents/cto_frames/registration_tmp/registrationV1/")
from registration_tmp.extractCenterline.lib.findroot import BinarySegment, FindRoot


# print(os.getcwd())

SRC_ROOT0 = r'/home/DataBase4/cto_gan_data3/LAO_test/merged'
dicoms_list = os.listdir(SRC_ROOT0)
# print(dicoms_list)

'''here1'''
seg_cls = BinarySegment()


def get_root(image, img_name, findroot_cls):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_result = seg_cls.inference(image)
    return findroot_cls.get_root(image_result, img_name)


#

for dicomj in dicoms_list:
    dicomi = os.path.join(SRC_ROOT0, dicomj)
    j=str(dicomj.split('m')[-1])
    # dicom_li=['1', '2', '3', '5', '6', '7', '8', '9', '11', '16', '17', '18', '19', '21', '23', '24', '25', '26', '27', '28', '29', '32', '33', '35', '38', '39', '42', '43', '44', '45', '49', '51', '52', '53', '54', '57', '58', '59', '60', '62', '63', '64', '67', '68', '72', '73', '78', '80', '81', '83', '84', '85', '86', '87', '88', '91', '92', '94', '96', '98', '99', '100']
    dicom_li=['1','2']
    if j not in dicom_li:
        continue
    print(dicomi)

    frames_list = os.listdir(dicomi)
    pbar=tqdm(frames_list)
    for src_path in pbar:
        pbar.set_description("Processing %s" % src_path)
        if not src_path.endswith('.jpg') and not src_path.endswith('.dcm'):
            src_path = os.path.join(dicomi, src_path)
            # print(src_path)
            # cmd = f"python src2tree.py {src_path}"
            # os.system(cmd)
            '''here2'''
            # cmd = f"python align_bin1208.py {src_path}"
            # os.system(cmd)
            '''here1'''
            
            OUT_ROOT = os.path.join(src_path, 'res')
            if os.path.exists(OUT_ROOT):
                for cfile in os.listdir(OUT_ROOT):
                    cfiles=os.path.join(OUT_ROOT,cfile )

                    os.remove(cfiles)


            SRC_ROOT = src_path
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
            if not os.path.exists(RES_LocateRoot+'_seg'):
                os.makedirs(RES_LocateRoot+'_seg')

            print('locate root...')
            IMG_ROOT = src_path_ROOT
            FILE_PATH = res_file_path
            ROOT_OUT_ROOT = RES_LocateRoot

            if not os.path.exists(ROOT_OUT_ROOT):
                os.makedirs(ROOT_OUT_ROOT)
            if os.path.exists(FILE_PATH):
                os.remove(FILE_PATH)
            findroot_cls = FindRoot(OUT_ROOT=ROOT_OUT_ROOT)
            with open(FILE_PATH, 'w') as root_f:
                # img_path_list = glob.glob(IMG_ROOT + '/' + '*.png')
                img_path_list = os.listdir(IMG_ROOT)
                img_path_list = list(
                    filter(lambda x: ('.jpg' in x) or ('.png' in x),
                           img_path_list))
                img_path_list = [
                    os.path.join(IMG_ROOT, x) for x in img_path_list
                ]
                for img_path in img_path_list:
                    print(img_path)
                    img = np.array(Image.open(img_path))
                    img_name = os.path.basename(img_path).split('.')[1]
                    if len(img.shape) > 2 and img.shape[-1] == 3:
                        png_img = img[:, :, 0]
                    else:
                        png_img = img
                    root = get_root(png_img, img_name + '.png', findroot_cls)
                    root_f.writelines(img_name + ' ' + str(root) + '\n')
                    # print (img_name,root)
            print('done!')
            

# SRC_ROOT = r'./test1118'
# cmd = f"python src2tree.py {SRC_ROOT}"
# os.system(cmd)

# cmd = f"python align_bin1118.py {SRC_ROOT}"
# os.system(cmd)
'''
export PYTHONPATH=$PYTHONPATH:/home/wly/Documents/cto_frames
export PYTHONPATH=$PYTHONPATH:/home/wly/Documents/cto_frames/registration_tmp
export PYTHONPATH=$PYTHONPATH:/home/wly/Documents/cto_frames/registration_tmp/extractCenterline
export PYTHONPATH=$PYTHONPATH:/home/wly/Documents/cto_frames/registration_tmp/extractCenterline/lib
export PATH=/home/wly/anaconda3/bin:$PATH
'''
