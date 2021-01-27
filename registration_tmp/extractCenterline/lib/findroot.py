# -*- coding: utf-8 -*-
from skimage.morphology import medial_axis
import cv2
import numpy as np
import copy
from skimage import measure, color, morphology
import os

import torch
from net import pspnet
import time
from PIL import Image
import torchvision.transforms as transforms


# if labels.max() == 0:
#    pass
#  û������ ��������������ĵ��ܶ˵�
#  ����������������������ĵ��ܵ�


class BinarySegment:

    def __init__(self):
        mean = [0.4519803823907019, 0.45198038239070190, 0.4519803823907019]
        std = [0.10958348599760581, 0.10958348599760581, 0.10958348599760581]
        self.imagetransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std)
        ])

        model_path = "./ckpt/{}.pth".format("daoguanzuozhuganpsp101")
        # model_path = "./model/{}.pth".format("daoguanzuozhuganpsp101")

        # define CNN network
        self.net = pspnet.PSPNet(n_classes=4,
                            sizes=(1, 2, 3, 6),
                            psp_size=2048,
                            deep_features_size=1024,
                            backend='resnet101')
        self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.load_state_dict(torch.load(model_path)['state_dict'])
        self.net.eval()

    def recover_pic(self, label, file_save_name):
        img = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)

        value_list = [0, 255, 128, 64]
        for index, value in enumerate(value_list):
            b = np.where(label == index)
            img[(b)] = value

        # save_pic = Image.fromarray(img)
        # save_pic.save("./result/" + file_save_name)

        return img

    def get_predictions(self, output_batch):
        bs, c, h, w = output_batch.size()
        tensor = output_batch.data

        values, indices = tensor.cpu().max(1)
        indices = indices.view(bs, h, w)
        return indices

    def inference(self, image):
        with torch.no_grad():
            image = self.imagetransform(image)
            eval_input = torch.autograd.Variable(image.unsqueeze(0).cuda(), requires_grad=False)
            output = self.net(eval_input)
            output = output.cpu()
            pred = self.get_predictions(output)
            image_result = self.recover_pic(pred[0].numpy(), "result.png")

        # cv2.imshow("result", image_result)
        # cv2.waitKey()
        return image_result


class FindRoot:
    def __init__(self,OUT_ROOT='./findRoot_result/'):

        self.OUT_ROOT = OUT_ROOT

    def get_root(self, image,img_name):
        OUT_ROOT = self.OUT_ROOT
        if not os.path.exists(OUT_ROOT):
            os.makedirs(OUT_ROOT)
        daoguan = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
        mainves = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
        inter = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

        b = np.where(image == 255)
        daoguan[(b)] = 255
        # --------------0517:draw picture--------------
        # cv2.imwrite(OUT_ROOT+'daoguan_before_'+img_name,daoguan)
        # cv2.imwrite('./0517result/daoguan_before.png',daoguan)


        b = np.where(image == 128)
        mainves[(b)] = 255

        daoguan_area1 = measure.label(daoguan, connectivity=2)
        daoguan_area = measure.regionprops(daoguan_area1)
        max_area = 0
        for reg in daoguan_area:
            if reg.area > max_area:
                # print(reg.area)
                max_area = reg.area
        # remove the area which is smaller than threshold(remove noise area).
        dst = morphology.remove_small_objects(daoguan_area1, min_size=max_area / 5, connectivity=2)
        daoguan = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

        b = np.where(dst > 0)
        daoguan[(b)] = 255

        # --------------0517:draw picture--------------
        # cv2.imwrite(OUT_ROOT+'daoguan_after_'+img_name,daoguan)
        # cv2.imwrite('./0517result/daoguan_after.png',daoguan)
        # cv2.imshow('frame2',daoguan)
        # cv2.waitKey(0)

        # �ҵ��ܺ�������ӵ�λ�ã�����inter��
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if daoguan[i, j] == 255:  # and image[i,j,1] == 255 and image[i,j,2] == 255:
                    dx = [1, 1, 0, -1, -1, -1, 0, 1]
                    dy = [0, 1, 1, 1, 0, -1, -1, -1]
                    for orien in range(8):
                        if 0 <= i + dx[orien] < 512 and \
                                0 <= j + dy[orien] < 512 and \
                                image[i + dx[orien], j + dy[orien]] == 128:
                            inter[i, j] = 255
                            # ------0517:add below----
                            inter[i+ dx[orien], j+ dy[orien]] = 255

        # --------------0517:draw picture--------------
        # cv2.imwrite(OUT_ROOT+'inter_'+img_name,inter)
        # cv2.imwrite('./0517result/inter.png',inter)

        # cv2.imshow('frame2', inter)
        # cv2.waitKey(0)

        inter_board = measure.label(inter, connectivity=2)
        zxz = [0, 0]
        if inter_board.max() == 0:
            skel, distance = medial_axis(daoguan, return_distance=True)
            oriskel = copy.deepcopy(skel)

            result = []

            def dfs(skel, root, result):
                skel[root[0], root[1]] = False
                dx = [1, 1, 0, -1, -1, -1, 0, 1]
                dy = [0, 1, 1, 1, 0, -1, -1, -1]
                count = 0
                for i in range(8):
                    if 0 <= root[0] + dx[i] < 512 and 0 <= root[1] + dy[i] < 512 and skel[root[0] + dx[i], root[1] + dy[i]]:
                        dfs(skel, [root[0] + dx[i], root[1] + dy[i]], result)
                        count += 1
                if count == 0:
                    result.append(root)

            def find(skel, result):
                for i in range(512):
                    for j in range(512):
                        if skel[i, j]:
                            result.append([i, j])
                            dfs(skel, [i, j], result)

            find(skel, result)
            final = []
            for member in result:
                dx = [1, 1, 0, -1, -1, -1, 0, 1]
                dy = [0, 1, 1, 1, 0, -1, -1, -1]
                count = 0
                for i in range(8):
                    if 0 <= member[0] + dx[i] < 512 and 0 <= member[1] + dy[i] < 512 and oriskel[
                        member[0] + dx[i], member[1] + dy[i]]:
                        count += 1
                if count == 1:
                    final.append(member)

            mainves_label = measure.label(mainves, connectivity=2)

            ro = 0
            if mainves_label.max() == 0:
                min_dis = 1000 * 1000
                for point in final:
                    dis = (256 - point[0]) ** 2 + (256 - point[1]) ** 2
                    if dis < min_dis:
                        min_dis = dis
                        ro = point
            else:
                mainves_region = measure.regionprops(mainves_label)

                max_area = 0
                re = 0
                for region in mainves_region:
                    if region.area > max_area:
                        max_area = region.area
                        re = region.centroid
                min_dis = 1000 * 1000
                for point in final:
                    dis = (re[0] - point[0]) ** 2 + (re[1] - point[1]) ** 2
                    if dis < min_dis:
                        min_dis = dis
                        ro = point

            zxz = ro

        else:
            candi = measure.regionprops(inter_board)
            x, y, count = 0., 0., 0.
            # ��������
            for region in candi:
                x += region.centroid[0]
                y += region.centroid[1]
                count += 1
            x /= count
            y /= count
            # print x,y
            zxz = [int(round(x)), int(round(y))]

        if zxz == 0:
            return None

        # --------------0517:draw picture--------------
        frame = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        # frame = cv2.circle(frame,(zxz[1],zxz[0]),3,(0,0,255))
        # -----------1119---------
        tmp_loc = np.where(frame==255)
        catheter_img = np.zeros((frame.shape))
        catheter_img[tmp_loc] = 255
        # ---------1119 over------
        cv2.imwrite(os.path.join(OUT_ROOT,'root_result_'+img_name),catheter_img)
        # cv2.imwrite(os.path.join(OUT_ROOT,'root_result_'+img_name),frame)
        # cv2.imwrite('./0517result/root_result.png',frame)

        # frame = cv2.circle(image, (zxz[1], zxz[0]), 5, (0, 0, 255), -1)
        # print("x == {}, y == {}".format(zxz[1], zxz[0]))
        # cv2.imwrite("./result/root.png", frame)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        print (img_name,zxz)
        return zxz  # h, w


if __name__ == "__main__":
    image = cv2.imread("./result/result.png", False)
    cls = FindRoot()
    cls.get_root(image)