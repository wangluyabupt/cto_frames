import numpy as np
import os
os.chdir('/home/DataBase3/wrj/workplace/extractCenterline/')
import glob
import cv2
from skimage.morphology import medial_axis,skeletonize
from ipdb import set_trace
from lib.build_tree import vascular_tree_build
import matplotlib.pyplot as plt
import sys



def get_skel(img_bin):
    image = img_bin
    '''这里需要把像素值转到0，1两种；'''
    # data = np.true_divide(image, 255)
    data = image
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(data, return_distance=True)
    return skel, distance

def get_skel2(img_bin):
    data = img_bin
    skel = skeletonize(data)
    return skel

IMG_ROOT = './dataset/basnet/CRA'
ROOT_FILE_PATH = './result/root.txt'
OUT_ROOT = './result/basnet/CRA'

# -----------medial_axis------------
# COARSE_OUT_ROOT = os.path.join(OUT_ROOT,'coarse')
# FINE_OUT_ROOT = os.path.join(OUT_ROOT,'fine')

# -----------skeletonize-------------
COARSE_OUT_ROOT = os.path.join(OUT_ROOT,'coarse_skeletonize')
FINE_OUT_ROOT = os.path.join(OUT_ROOT,'fine_skeletonize')
COR_OUT_ROOT = os.path.join(OUT_ROOT,'coordinate')


pos = IMG_ROOT.split('/')[-1]

for pth in [OUT_ROOT,COARSE_OUT_ROOT,FINE_OUT_ROOT,COR_OUT_ROOT]:
    if not os.path.exists(pth):
        os.makedirs(pth)

root_dic = {}
with open(ROOT_FILE_PATH,'r') as root_f:
    for line in root_f.readlines():
        line = line.strip('\n')
        if '[' in line:
            line = line.split('[')
            name,root = line[0].strip(' '), str('[') + line[1]
        else:
            line = line.split(' ')
            name, root = line[0], line[1]

        root_dic[name] = eval(root)

bin_img_name_list = os.listdir(IMG_ROOT)

all_invalid_imgs = []
all_failed_imgs = []
for img_name in bin_img_name_list:
    print (img_name)
    # for pix2pix
    # query = img_name.replace('_synthesized_image.jpg','')
    # for basnet
    query = img_name.split('.')[0]
    root_point = root_dic[query]
    if not root_point:
        print('No valid root, skip!')
        continue
    # root_point = [115,277] # [Y,X] [124,237] [85,256]

    img_path = os.path.join(IMG_ROOT,img_name)
    out_skel_path = os.path.join(COARSE_OUT_ROOT,'coarse_skel_'+ img_name)
    ori_img = cv2.imread(img_path)
    tmp_loc = np.where(ori_img > 128)
    new_img = np.zeros_like(ori_img)
    new_img[tmp_loc] = 255
    if len(new_img.shape) > 2 and new_img.shape[-1] == 3:
        new_img = new_img[:, :, 0]
    # cv2.imwrite(out_img_path,new_img)
    # ----------------medial_axis-------------
    skel, dis = get_skel(new_img / 255)
    dist_on_skel = skel * dis
    # ----------skeletonize-----------
    skel2 = get_skel2(new_img / 255)
    cv2.imwrite(out_skel_path, skel2*255)

    # ---------------visualize for comparing medial_axis and skeletonize---------------------
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
    #                          sharex=True, sharey=True)
    # ax = axes.ravel()
    #
    # ax[0].imshow(skel*255, cmap=plt.cm.gray)
    # ax[0].set_title('medial_axis')
    # ax[0].axis('off')
    #
    # ax[1].imshow(skel2*255, cmap=plt.cm.gray)
    # ax[1].set_title('skeletonize')
    # ax[1].axis('off')
    #
    # fig.tight_layout()
    # plt.show()
    # -------------visualization over--------------

    # cv2.imshow('skel2',show_skel2)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # cv2.imwrite(out_skel_path, skel * 255)

    # --------optimize skel-------------
    img_color = np.zeros((new_img.shape[0], new_img.shape[1], 3))
    cls_build_tree = vascular_tree_build.VascularTreeBuild()

    # 1. locate the real root point
    tmp_loc = np.where(skel2 > 0)
    min_dist = 100
    real_root = None
    root_point = np.array(root_point)
    root_flag = 0
    for i in range(len(tmp_loc[0])):
        coor = np.array([tmp_loc[0][i],tmp_loc[1][i]])
        tmp_dist = np.linalg.norm(coor-root_point)
        if tmp_dist < min_dist:
            min_dist = tmp_dist
            real_root = coor
            root_flag = 1
    if not root_flag:
        print("Can't locate the real root!")
        continue
    real_root = list(real_root)

    # 2. build tree
    root_node, branch_nodes = cls_build_tree.main(real_root,
                                                  skel2, # skeletonize
                                                  dis, # Mind: generated by medial_axis, but is not used in the function
                                                  img_color,
                                                  POSITION=pos)
    tree_nodes = root_node
    coordinates = []
    bifurcation = []
    end_point = []

    if tree_nodes is not None:
        print("树总长度为: {}".format(tree_nodes._all_child_count))
        if tree_nodes._all_child_count < 100:
            all_invalid_imgs.append(img_name)
        img_for_show = np.zeros([512, 512, 3], dtype=np.uint8)
        node_set = [tree_nodes]
        # print('note_set ::', node_set)
        i = 0
        while len(node_set):
            tree_nodes = node_set.pop()  #
            # print('tree_nodes ::', tree_nodes)

            while (tree_nodes):
                h, w = tree_nodes._value[0:2]
                i += 1
                # 注意 OpenCV 中的坐标顺序为 (w,h), numpy数组所表示的图像坐标顺序为(h,w)
                img_for_show[h][w] = [255, 255, 255]
                child_count = len(tree_nodes._child)
                if child_count:
                    if child_count > 1:
                        # print("找到一个二分叉")
                        # 注意 OpenCV 中的坐标顺序为 (w,h), numpy数组所表示的图像坐标顺序为(h,w)
                        bifurcation.append([w, h])
                        for id in range(1, child_count, 1):
                            node_set.append(tree_nodes._child[id])  #
                    tree_nodes = tree_nodes._child[0]
                else:
                    end_point.append([w, h])
                    tree_nodes = 0
                coordinates.append([w, h])
        img_skel_save_path = os.path.join(FINE_OUT_ROOT, 'skel_' + img_name)
        cv2.imwrite(img_skel_save_path, img_for_show)
        print('successfully!')
    else:
        print("failed!!!")
        all_failed_imgs.append(img_name)
        continue

    # print('coordinates ::', coordinates)
    file_path = os.path.join(COR_OUT_ROOT, img_name.split('.')[0] + '.txt')
    with open(file_path, 'w') as f:
        f.write(str(coordinates))

print('failed_imgs:\n',all_failed_imgs)
print('invalid_imgs:\n',all_invalid_imgs)
