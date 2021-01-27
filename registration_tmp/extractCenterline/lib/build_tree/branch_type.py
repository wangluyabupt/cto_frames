# coding=utf-8

import numpy as np
import copy

__all__ = [
    'BranchType',
]

'''
    # 本类用于进行分枝处的各种判断；
    # 判断血管类型:主干/旁枝；
'''


class BranchType:
    def __init__(self, img_seg, CoronaryPosition="RIGHT"):
        self.img_seg = img_seg
        self.CoronaryPosition = CoronaryPosition

    def __del__(self):
        pass

    # @staticmethod
    def bi_branch_node_compose(self, node):
        '''
        # 该函数是实际判断二分叉节点的哪个孩子节点是主支的处理函数；
        # 该函数传入一个分叉节点，组合三个方向的分支段；

        Args:
            node: 传入一个需要进行判断的树节点；

        Returns:
            返回主干孩子节点的序号，如0, 1，分别代表0孩子节点是主干或是1孩子节点是主干；
        '''

        main_father = []
        son_0 = []
        son_1 = []

        # 找当前点的前10个父节点;
        tmp_node = node._pre_node

        # 使用循环提前到当前点向前第10个像素点；
        for i in range(1, 10):  # 找当前节点的前向节点，也就是当前节点的前10个父节点；
            if tmp_node == None:
                print("ERROR--in StenosisMeasureLib--Tree--branch_type.py--BranchType--bi_branch_node_compose--判断主孩子节点时，添加传入节点的父节点失败，父节点为None")
                print("ERROR :: src bi_node :: ", node._value)
                return None
            else:
                main_father.append([tmp_node._value[0], tmp_node._value[1]])
                if tmp_node._pre_node is not None:
                    tmp_node = tmp_node._pre_node  # 这里应当进行判断才是正确的；
                else:
                    break

        # 找当前点0孩子的后10个点；
        left_main_color_count = 0
        right_main_color_count = 0
        tmp_node = node._child[0]

        for i in range(1, 10):  # 添加当前传入节点0孩子的后10个节点；
            if tmp_node == None:
                print("ERROR--in StenosisMeasureLib--Tree--branch_type.py--BranchType--bi_branch_node_compose--判断主孩子节点时，添加传入节点的0孩子节点失败，0孩子节点为None")
                return None
            else:
                son_0.append([tmp_node._value[0], tmp_node._value[1]])
                if len(tmp_node._child):
                    tmp_node = tmp_node._child[0]  # 这里应当进行判断才是正确的；
                else:
                    break
                # 判断后续子节点有多少个主干颜色；
                [r, g, b] = self.img_seg[tmp_node._value[0]][tmp_node._value[1]]  # 这里的node里边存储的下标0，1对应的就是h,w；目前应该是标准的RGB-Order;

                # 修改后新增前降支远[255, 255, 0]，删除一个左主干[255, 0, 0]
                if [r, g, b] in [[255, 128, 0], [255, 0, 128], [0, 255, 128], [255, 0, 0], [0, 255, 0], [255, 0, 255],
                                 [255, 255, 0], [0, 0, 128], [0, 128, 0]]:
                    left_main_color_count += 1

        # 找当前点1孩子的后10个点;
        tmp_node = node._child[1]
        for i in range(1, 10):
            if tmp_node == None:
                print("ERROR--in StenosisMeasureLib--Tree--branch_type.py--BranchType--bi_branch_node_compose--添加传入节点的1孩子节点失败，1孩子节点为None")
                return None
            else:
                son_1.append([tmp_node._value[0], tmp_node._value[1]])
                if len(tmp_node._child):
                    tmp_node = tmp_node._child[0]  # 这里应当进行判断才是正确的；
                else:
                    break

                [b, g, r] = self.img_seg[tmp_node._value[0]][tmp_node._value[1]]  # 这里的node里边存储的下标0，1对应的就是h,w；

                # 同上，修改后新增前降支远[255, 255, 0]，删除一个左主干[255, 0, 0]
                if [r, g, b] in [[255, 128, 0], [255, 0, 128], [0, 255, 128], [255, 0, 0], [0, 255, 0], [255, 0, 255],
                                 [255, 255, 0], [0, 0, 128], [0, 128, 0]]:
                    # left_main_color_count += 1
                    right_main_color_count += 1

        # 这里申请空间的原因是，当前点的前10个，当前点的后10个，重新组合在一起；
        # line_0 = [None] * 21  # f and son_0
        # line_1 = [None] * 21  # f and son_1
        # line_2 = [None] * 21  # son_0 and son_1
        line_0 = [None] * 3
        line_1 = [None] * 3
        line_2 = [None] * 3

        # tmp_father = main_father[::-1]
        # line_0[0:10] = copy.deepcopy(tmp_father)
        # line_0[10] = node._value
        # line_0[11:] = copy.deepcopy(son_0)
        line_0[0] = main_father[-1]
        line_0[1] = node._value[0:2]
        line_0[2] = son_0[-1]

        # line_1[0:10] = copy.deepcopy(tmp_father)
        # line_1[10] = node._value
        # line_1[11:] = copy.deepcopy(son_1)
        line_1[0] = main_father[-1]
        line_1[1] = node._value[0:2]
        line_1[2] = son_1[-1]

        # line_2[0:10] = copy.deepcopy(son_0)
        # line_2[10] = node._value
        # line_2[11:] = copy.deepcopy(son_1)
        line_2[0] = son_0[-1]
        line_2[1] = node._value[0:2]
        line_2[2] = son_1[-1]

        # 调用角度计算函数；
        degree_0 = self.main_branch_detection(line_0)
        degree_1 = self.main_branch_detection(line_1)
        degree_2 = self.main_branch_detection(line_2)

        # 对主干的判断还需要继续；
        # 否则其它的都按照角度以及参考直径的大小来进行区分；
        # 如果是Y型分叉，选择左边的；
        # 二级分叉的后两位直接随机；
        # if degree_0 < degree_1:
        # print(
        #     "left_main_color_count = %d, right_main_color_count = %d" % (left_main_color_count, right_main_color_count))
        if self.CoronaryPosition is not "RIGHT":
            if left_main_color_count > right_main_color_count:
                # print("0孩子是主干")
                # 把所有0孩子的颜色进行标记；
                # for tmp_point in son_0:
                # h = tmp_point[0]
                # w = tmp_point[1]
                # img_binary_for_show[h][w] = [255,255,0]
                # cv2.circle(self.img_for_medina, (son_0[-1][1], son_0[-1][0]), 5, (0, 255, 0), 1)
                # main_son = copy.deepcopy(son_0)
                # branch_son = copy.deepcopy(son_1)

                return 0
            #
            else:
                # print("1孩子是主干")
                # cv2.circle(self.img_for_medina, (son_1[-1][1], son_1[-1][0]), 5, (0, 255, 0), 1)
                # main_son = copy.deepcopy(son_1)
                # branch_son = copy.deepcopy(son_0)

                return 1

        # 这里返回的角度，应当是越大越是主干的延续；
        if degree_0 < degree_1:
            # print("0孩子是主干")
            # 把所有0孩子的颜色进行标记；
            # for tmp_point in son_0:
            # h = tmp_point[0]
            # w = tmp_point[1]
            # img_binary_for_show[h][w] = [255,255,0]
            # cv2.circle(self.img_for_medina, (son_0[-1][1], son_0[-1][0]), 5, (0, 255, 0), 1)
            main_son = copy.deepcopy(son_0)
            branch_son = copy.deepcopy(son_1)

            # return 0
            return 1
        #
        else:
            # print("1孩子是主干")
            # cv2.circle(self.img_for_medina, (son_1[-1][1], son_1[-1][0]), 5, (0, 255, 0), 1)
            main_son = copy.deepcopy(son_1)
            branch_son = copy.deepcopy(son_0)

            # return 1
            return 0

    # 这里是计算传入的两根直线之间的夹角，并返回；
    def angle_calculation(self, points_a, points_b):
        '''
        
        
        '''
        if points_a[0][1] != points_a[-1][1]:
            k1 = (points_a[0][0] - points_a[-1][0]) / (points_a[0][1] - points_a[-1][1])
            k1_flag = True
        else:
            k1_flag = False
        if points_b[0][1] != points_b[-1][1]:
            k2 = (points_b[0][0] - points_b[-1][0]) / (points_b[0][1] - points_b[-1][1])
            k2_flag = True
        else:
            k2_flag = False

        #
        if k1_flag and k2_flag:
            if k1 * k2 == -1:
                # cv2.circle(self.img_for_medina,(points_a[5][1],points_a[5][0]),10,(0,0,255),1)
                # print("找到一个直角")
                degree = 90
            else:
                degree = np.arctan(np.abs((k2 - k1) / (1 + k1 * k2))) * (180 / np.pi)
        else:
            # 如果有一个是垂直或是两个都是垂直的；
            if not k1_flag and k2_flag:
                degree = 90 - np.arctan(np.abs(k2)) * (180 / np.pi)
                # print("k1垂直，k2不垂直")
            elif k1_flag and not k2_flag:
                degree = 90 - np.arctan(np.abs(k1)) * (180 / np.pi)
                # print("k2垂直，k1不垂直")
            elif not k1_flag and not k2_flag:
                degree = 0
                # print("两条线都垂直，夹角为0")

        return degree

    # 传入所有二分叉的节点数据；
    # 返回对分叉点处的判断；
    def branch_type(self, bifurcate_points, branch_points, root_node):
        '''

        '''
        self.__tmp_segment = []
        self.__all_segment = []

        #
        if bifurcate_points == None or branch_points == None:
            print("in BranchType :: branch_type ::　传入的二分叉点集或是分支点集为空")

        # 传入整段的数据，进行重新的拆分；
        self.__tmp_head_set = [root_node]

    def main_branch_detection(self, points):
        """
        这里是计算两边的夹角，推荐使用三角函数来进行计算；

        Args:
            points:

        Returns:

        """

        # head_point = points[0]
        # mid_point = points[10]
        # tail_point = points[-1]
        head_point = points[0]
        mid_point = points[1]
        tail_point = points[-1]

        a = np.sqrt((mid_point[0] - head_point[0])**2 + (mid_point[1] - head_point[1])**2)
        b = np.sqrt((mid_point[0] - tail_point[0]) ** 2 + (mid_point[1] - tail_point[1]) ** 2)
        c = np.sqrt((tail_point[0] - head_point[0]) ** 2 + (tail_point[1] - head_point[1]) ** 2)
        degree = np.degrees(np.arccos((a**2 + b**2 - c**2) / (2*a*b)))
        # print("in branch_type--main_branch_detection--degree: {}".format(degree))

        # 返回计算出来的角度；
        return degree

    def main(self, node):
        # 主函数:传入需要判断主孩子节点的二分叉节点；
        # 返回主分支序号；
        main_branch_nums = self.bi_branch_node_compose(node)

        return main_branch_nums
