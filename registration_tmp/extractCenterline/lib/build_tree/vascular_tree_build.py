# coding=utf-8

import numpy as np
import sys
import cv2
import copy
import datetime

from lib.build_tree.branch_type import BranchType

# python默认的递归深度好像是1000，这里需要进行手动设置；
sys.setrecursionlimit(10000000)
'''
该代码用于在提取的骨架图上建树；
'''


class node:
    """
    树节点定义类；
    """

    def __init__(self, value):
        self._value = value  # 存储h,w,dist;
        self._child = []  # 孩子节点；
        self._child_count = 0  # 孩子节点数目；
        self._pre_node = None  # 父节点；
        self._all_child_count = 0  # 当前子节点到叶节点的所有节点数目；
        self._vertical_line_points = []  # 当前骨架点与血管边缘垂线坐标点，示例[[h, w], [h, w]] == [[y, x], [y, x]];


class VascularTreeBuild:
    """
    血管树建树类；
    """

    def __init__(self, H=512, W=512, threshold=20):
        self.H = H  # 默认这里的处理尺寸为512；
        self.W = W
        self.threshold = threshold  # 对分支是否有效的长度进行判断，该阈值存在的意义主要是去掉毛刺以及一些可能由于分割效果导致的不正确的骨架；
        self._terminal_points = []

    def branch_main_check(self, child_set):
        """
        传入待选的树节点坐标，对其进行主干节点的判断；
        并进行重新组织成主干在前的形式；

        Args:
            child_set:  # 这里传入的是竟然只是坐标，而不是树节点；

        Returns:
            这里的返回值竟然也只是坐标点，而不是树节点；
        """

        tmp_avg_width_record = []
        search_len = 10  # 设置向前预取点进行计算的个数；
        for i in child_set:  # 对传入的所有分支节点每一个点进行遍历；
            all_width = 0.0
            avg_width = 0.0

            # 起始节点赋值；
            # 需要注意对tmp_h以及tmp_w进行更新；
            tmp_h, tmp_w = i[0], i[1]
            tt_child_set = []

            # 前看的点数控制；
            ttmp_record = copy.deepcopy(self.img_record)
            valid_child_count = 0
            for t in range(0, search_len):
                # 找其八领域，未访问的点，进行访问；
                # 因为每一次开始进行访问的点不一样，所以这里需要进行一个深拷贝的操作; 
                child_count = 0
                for h_c in range(-1, 2):
                    for w_c in range(-1, 2):
                        n_h = tmp_h + h_c
                        n_w = tmp_w + w_c
                        # 先排队当前坐标点；
                        # 判断坐标的合法性；
                        # 判断相邻坐标点可访问性；
                        if (not (h_c == 0 and w_c == 0)
                                and n_h >= 0 and n_w >= 0
                                and n_h < self.H
                                and n_w < self.W
                                and self.img_skel[n_h][n_w]
                                and not ttmp_record[n_h][n_w]):
                            # 在临时的点这里进行访问记录；
                            # print("找到一个合法点")
                            ttmp_record[n_h][n_w] = 1
                            child_count += 1
                            # 进入递归，如果要进行控制，需要在递归前就进行处理；
                            tt_child_set.append([n_h, n_w])

                # check and update;
                if child_count > 1 or child_count == 0:
                    # print("already reach branch point or end point")
                    pass
                else:
                    all_width += self.distance[tt_child_set[0][0], tt_child_set[0][1]]
                    valid_child_count += 1
                    pass
                # tmp_h = tt_child_set[child_count-1][0]
                # tmp_w = tt_child_set[child_count-1][1]
            if valid_child_count:
                avg_width = all_width / valid_child_count
                tmp_avg_width_record.append(avg_width)
                # print("avg_width == ",avg_width)
            else:
                # print("valid_child_count == ",valid_child_count)
                tmp_avg_width_record.append(0)

        # 所有孩子节点下看search_len个节点后，计算的平均宽度，来进行判断；
        # 是否有必要把所有分支按照由粗到细的大小来进行排序；
        # 还是只把最粗的分支放到第0个位置来；
        # 下面进行简单的交换；
        max_index = tmp_avg_width_record.index(max(tmp_avg_width_record))
        ttt_tmp_h = child_set[0][0]
        ttt_tmp_w = child_set[0][1]
        child_set[0][0] = child_set[max_index][0]
        child_set[0][1] = child_set[max_index][1]
        child_set[max_index][0] = ttt_tmp_h
        child_set[max_index][1] = ttt_tmp_w

        return child_set

    def bi_nodes_check(self, node):
        """
        该函数用于对主干内存在的二分叉节点的有效性进行判断；
        由于血管树根节点参考点无法应对各种情况，存在部分二分叉无效的情况，需要对这类二分叉有效性进行判断；


        判断方法：
        1. 即将是添加的第一个二分叉节点；
        2. 当前点在主干区域；
        3. 有一个分支长度不长，孩子节点总数相对极少；
        4. 所有孩子节点全在主干区域；

        self._value = value  # 存储h,w,dist;
        self._child = []  # 孩子节点；
        self._child_count = 0  # 孩子节点数目；
        self._pre_node = None  # 父节点；
        self._all_child_count = 0  # 当前子节点到叶节点的所有节点数目；
        self._vertical_line_points = []  # 当前骨架点与血管边缘垂线坐标点，示例[[h, w], [h, w]] == [[y, x], [y, x]];

        """

        if len(self._branch_nodes) != 0:  # 通过数量来判断 是否为需要进行处理的二分叉；
            # print("当前二分叉节点集合已有分叉节点，不符合判断标准")
            return [True, None]

        # 对分叉存在位置进行判断；
        h, w = node._value[0:2]
        [r, g, b] = self.img_color[h][w]
        MAIN_BRANCH_COLOR = [255, 0, 0]  # 左冠主干应该都是红色；
        if self.POSITION == "RIGHT":
            MAIN_BRANCH_COLOR = [255, 128, 0]

        if not np.array_equal([r, g, b], MAIN_BRANCH_COLOR):  # 当前分叉位置不在主干内；
            return [True, None]

        # 对短的分支所有节点是否在主干区域进行判断；
        child_nums_set = []
        for child in node._child:
            child_nums_set.append(child._all_child_count)

        index = np.argmin(child_nums_set)
        child_node = node._child[index]  # 找到最短的孩子节点；
        MAIN_COLOR_COUNT = 0
        NOT_MAIN_COLOR_COUNT = 0
        while child_node is not None:
            h, w = child_node._value[0:2]
            [r, g, b] = self.img_color[h][w]
            if np.array_equal([r, g, b], MAIN_BRANCH_COLOR):  # 当前分叉位置不在主干内；
                MAIN_COLOR_COUNT += 1
            else:
                NOT_MAIN_COLOR_COUNT += 1
            # print("child_node._child length: {}".format(len(child_node._child)))
            if len(child_node._child):
                # 这里一定要记得更新，这里默认的考虑是假设这里后续也没有其它分支了；
                # 这种考虑无法保证完全正确；
                child_node = child_node._child[0]
            else:
                child_node = None

        # 当在主干区域的节点数要大于非主干节点数时，认为当前二分叉节点是无效的，应当去掉短的那一支；
        if MAIN_COLOR_COUNT > NOT_MAIN_COLOR_COUNT:
            return [False, index]

        return [True, None]

    def get_valid_neighbors(self, H, W):
        """
        该函数用于对传入坐标查找八领域合法像素；
        合法像素包括有骨架上，同时没有被访问过；

        """

        neightbor_set = []
        tmp_terminal_set = []  # 对传入坐标的八邻域进行判断，把符合的添加进来；
        for i in range(-1, 2):
            for j in range(-1, 2):
                n_h = H + i
                n_w = W + j
                if not (i == 0 and j == 0) and \
                        n_h >= 0 and n_w >= 0 and \
                        n_h < self.H and \
                        n_w < self.W and \
                        self.img_skel[n_h][n_w]:  # 坐标合法性检测，在这里要判断是否在骨架点上；

                    if not self.img_record[n_h][n_w]:  # 如果当前点还没有被遍历到；
                        self.img_record[n_h][n_w] = 1  # 对当前已经遍历的点进行记录；
                        neightbor_set.append([n_h, n_w])  # 进入递归，如果要进行控制，需要在递归前就进行处理；

                    tmp_terminal_set.append([n_h, n_w])  # 对这里表示怀疑，这里是添加当前传入节点的所有相邻节点到set中，相当于是判断当前节点有几个相邻点；

        return neightbor_set

    def get_tree_without_recursive(self, root_node):
        """
        本函数是使用非递归的方式来建树，用于缩短建树中使用递归的时间消耗；
        可能涉及到广度优先和深度优先的选择，深度优先的弊端以前也遇到过，当遇到环的时候，可能不知道怎么打破，广度优先当然也无法解决这个问题；
        如果选择广度优先，之前的一些对当前节点的有效性判断可能就无法进行，

        Args:
            root_node:

        Returns:

        """

        DFS_VECTOR = []  # 深度优先搜索向量；
        # DFS_VECTOR.append(copy.deepcopy(root_node))  # 存入第一个可以开始的点，一定要注意list默认是对象的引用；
        DFS_VECTOR.append(root_node)  # 存入第一个可以开始的点，一定要注意list默认是对象的引用；
        while len(DFS_VECTOR):  # 当这里vector为空，则表明整个处理结束；
            TMP_NODE = DFS_VECTOR.pop(0)  # 这里pop默认是第0个，list的pop操作会返回被pop的值；
            c_h, c_w = TMP_NODE._value[0:2]

            CURRENT_NODE = TMP_NODE
            while (1):  # 对当前节点进行DFS处理；
                self.count += 1
                tmp_child_points_set = []
                # tmp_terminal_points_set = []  # 对传入坐标的八邻域进行判断，把符合的添加进来；
                for h_c in range(-1, 2):
                    for w_c in range(-1, 2):
                        n_h = c_h + h_c
                        n_w = c_w + w_c
                        if not (h_c == 0 and w_c == 0) and \
                                n_h >= 0 and n_w >= 0 and \
                                n_h < self.H and \
                                n_w < self.W and \
                                self.img_skel[n_h][n_w]:  # 坐标合法性检测，在这里要判断是否在骨架点上；

                            if not self.img_record[n_h][n_w]:  # 如果当前点还没有被遍历到；
                                self.img_record[n_h][n_w] = 1  # 对当前已经遍历的点进行记录；
                                tmp_child_points_set.append([n_h, n_w])  # 进入递归，如果要进行控制，需要在递归前就进行处理；

                # 只有分支点处才需要check
                tmp_len = len(tmp_child_points_set)
                # print("tmp_len: {}".format(tmp_len))
                if tmp_len == 0:  # 如果没有有效的领域节点，则表明当前DFS结束；
                    break

                if tmp_len > 1:  # 如果有多个，则应当选择主干孩子节点进行继续DFS，而把非主的孩子节点加入最顶层的DFS_VECTOR中；并且这里应当创建node，并进行添加多个孩子；
                    tmp_child_points_set = self.branch_main_check(
                        tmp_child_points_set)  # 判断孩子节点哪一个是主干，优先考虑递归主干孩子节点，这里的判断方法有问题，如果主干发生狭窄则失效；
                elif tmp_len == 1:
                    # 把terminal points存储起来，要注意不同分支的不同情况；
                    # self._terminal_points.append(tmp_terminal_points_set[0])  # 对这里的terminal_points表示怀疑，建议不使用这里的数据结果；
                    pass

                # 把每个新的领域添加到本节点的子节点中；
                for point in tmp_child_points_set:
                    h, w = point
                    tmp_node = node([h, w])  # 生成一个新节点；
                    tmp_node._pre_node = CURRENT_NODE  # 更新当前节点的参数；
                    # CURRENT_NODE._child.append(copy.deepcopy(tmp_node))
                    CURRENT_NODE._child.append(tmp_node)
                    CURRENT_NODE._child_count += 1  # TODO 暂时无法更新；似乎这两个数据在分叉处才有实际意义，用于判断是舍还是留；
                    # CURRENT_NODE._all_child_count += tt_count + 1  # TODO 暂时无法更新；

                if CURRENT_NODE._child_count > 1:  # 对孩子节点处理后再进行判断是否是分叉节点；
                    # self._branch_nodes.append(copy.deepcopy(CURRENT_NODE))
                    self._branch_nodes.append(CURRENT_NODE)
                    DFS_VECTOR.extend(CURRENT_NODE._child[1:])

                # 上边的tmp_child_set，并不是树节点，而只是坐标点，所以后续的处理这样使用会导致失败；
                # 这里有一点，这里不一定有孩子节点，在前边判断到当前为终端节点的时候，应该结束当前的DFS进程；
                CURRENT_NODE = CURRENT_NODE._child[0]
                c_h, c_w = CURRENT_NODE._value[0:2]  # 一定要注意更新这里的坐标，否则无法继续；



        return root_node

    def get_tree(self, root_node, c_h, c_w):
        """
        传入一个根结点以及起始坐标，返回一颗树

        Args:
            root_node: 传入的空树节点；
            c_h:
            c_w:

        Returns:

        """

        tmp_child_set = []
        tmp_terminal_set = []  # 对传入坐标的八邻域进行判断，把符合的添加进来；
        for h_c in range(-1, 2):
            for w_c in range(-1, 2):
                n_h = c_h + h_c
                n_w = c_w + w_c
                if not (h_c == 0 and w_c == 0) and \
                        n_h >= 0 and n_w >= 0 and \
                        n_h < self.H and n_w < self.W and \
                        self.img_skel[n_h][n_w]:  # 坐标合法性检测；

                    if not self.img_record[n_h][n_w]:  # 如果当前点还没有被遍历到；
                        self.img_record[n_h][n_w] = 1  # 对当前已经遍历的点进行记录；
                        tmp_child_set.append([n_h, n_w])  # 进入递归，如果要进行控制，需要在递归前就进行处理；

                    tmp_terminal_set.append([n_h, n_w])  # 对这里表示怀疑，这里是添加当前传入节点的所有相邻节点到set中，相当于是判断当前节点有几个相邻点；


        # 只有分支点处才需要check
        tmp_len = len(tmp_child_set)
        if tmp_len > 1:
            # 传入一所有孩子节点的list，返回一个对应大小的list;
            tmp_child_set = self.branch_main_check(tmp_child_set)  # 判断孩子节点哪一个是主干，优先考虑递归主干孩子节点；
        elif tmp_len == 1:
            # 把terminal points存储起来，要注意不同分支的不同情况；
            self._terminal_points.append(tmp_terminal_set[0])  # 对这里的terminal_points表示怀疑，建议不使用这里的数据结果；

        # 把每个新的邻域添加到本节点的子节点中；
        for t in tmp_child_set:
            n_h, n_w = t
            tmp_node = node([n_h, n_w])  # 生成一个新节点；
            tt_node, tt_count = self.get_tree(tmp_node, n_h, n_w)  # 以该节点进行递归；
            ##############################################################
            # 其实是可以直接在这里选择哪些子节点是可以添加 哪些是不可以添加的；
            # if tt_node._all_child_count < self.threshold:
            # if tt_count < self.threshold:
            #     print("当前新节点长度不满足要求，将会被删除: {}".format(tt_count))
            #     # del tmp_node
            #     continue
            ##############################################################
            tmp_node._pre_node = root_node  # 更新当前节点的参数；
            root_node._child.append(tmp_node)
            root_node._child_count += 1
            root_node._all_child_count += tt_count + 1  #

        tmp_len = len(root_node._child)
        # 判断当前节点是否是分叉点，并根据子分支的孩子节点总数多少来判断是否删除;
        if tmp_len > 1:  # 这里是只对有多个孩子节点的情况进行处理； #
            index_set = []
            #
            child_dist_set = []  # 存储每个孩子到根节点的距离；
            child_count_set = []  #存储每个孩子的总孩子节点数量；
            child_index_set = []  # 存储对应孩子节点的索引；
            for i in range(0, tmp_len):
                t_c = root_node._child[i]
                t_h, t_w = root_node._child[i]._value[0:2]
                t_dist = np.sqrt((self.all_head_points[0] - t_h)**2 + (self.all_head_points[1] - t_w)**2)
                # print("当前分叉点到根节点的距离为: {}".format(t_dist))
                # 对每个孩子的分支长度进行判断；
                if t_c._all_child_count < self.threshold: #  or t_dist < 5:  # 这里不应当用这样简单的方法进行判断，两个孩子距离都小，但有一个是在长度上满足的；
                    index_set.append(i)  # 添加不符合长度的分支序号；

                if t_dist < 3:  # 分叉点到根节点的距离阈值，小于此阈值，成为无效候选子节点；
                    child_dist_set.append(t_dist)
                    child_count_set.append(root_node._child[i]._all_child_count)
                    child_index_set.append(i)

            # 在这里集中对根节点处非法的分叉点进行处理；
            if len(child_index_set):
                max_index = np.argmax(child_count_set)
                print("非法分叉点的最大子节点序号: {}".format(max_index))
                print("child_index_set: {}".format(child_index_set))
                # child_index_set.remove(max_index)  # 直接保留最大的孩子节点，将其它节点都删除；
                child_index_set.pop(max_index)
                index_set.extend(child_index_set)

            # 这里不应当在for循环以内，否则可能会删除掉所有点；
            index_set = list(set(index_set))  # 去掉重复的索引，否则会出错；
            nodes_check = self.bi_nodes_check(root_node)  # 对当前的二分叉节点进行检查，排除不合理的二分叉；
            if not nodes_check[0] and (nodes_check[1] not in index_set):  # 避免对同一个节点的重复添加；
                # print("当前二分叉节点为无效节点，将被删除!")
                # tmp_img_for_show = copy.deepcopy(self.img_color)
                # cv2.circle(tmp_img_for_show, (c_w, c_h), 10, (0, 255, 0), -1)
                # cv2.imshow("bi node check", tmp_img_for_show)
                # cv2.waitKey()
                index_set.append(nodes_check[1])

            # print("以下序号节点将会被删除: {}".format(index_set))
            # print("all child count: {}".format(root_node._child_count))
            # print("index_set: {}".format(index_set))
            if len(index_set):
                root_node._child_count -= len(index_set)  # 更新孩子节点数量；
                for k in range(len(index_set)):  # 删除长度小于阈值的分支；
                    root_node._all_child_count -= (
                                root_node._child[index_set[k] - k]._all_child_count + 1)  # 更新当前节点后续所有孩子节点的数量；

                    del root_node._child[index_set[k] - k]  # 删除对应序号的孩子节点；

        if root_node._child_count > 1:  # 对孩子节点处理后再进行判断是否是分叉节点；
            self._branch_nodes.append(root_node)

        return root_node, root_node._all_child_count

    def get_tree_manual_skel_lxh(self, root_node, c_h, c_w):
        """
        传入一个根结点以及起始坐标，返回一颗树

        Args:
            root_node: 传入的空树节点；
            c_h:
            c_w:

        Returns:

        """

        tmp_child_set = []
        tmp_terminal_set = []  # 对传入坐标的八邻域进行判断，把符合的添加进来；
        print('root node ::', root_node)
        for h_c in range(-1, 2):
            for w_c in range(-1, 2):
                n_h = c_h + h_c
                n_w = c_w + w_c
                r, g, b = self.img_skel[n_h][n_w]
                print(r, g, b)

                if (not (h_c == 0 and w_c == 0)) and \
                        n_h >= 0 and n_w >= 0 and \
                        n_h < self.H and n_w < self.W and \
                        (r != 0) or (b != 0) or (g != 0):  # 坐标合法性检测；

                    if not self.img_record[n_h][n_w]:  # 如果当前点还没有被遍历到；
                        self.img_record[n_h][n_w] = 1  # 对当前已经遍历的点进行记录；
                        tmp_child_set.append([n_h, n_w])  # 进入递归，如果要进行控制，需要在递归前就进行处理；

                    tmp_terminal_set.append([n_h, n_w])  # 对这里表示怀疑，这里是添加当前传入节点的所有相邻节点到set中，相当于是判断当前节点有几个相邻点；


        # 只有分支点处才需要check
        tmp_len = len(tmp_child_set)
        print('tmp length ::', tmp_len)
        # 手动将前降支放在首位，回旋支放在第二位。只针对左主干分叉的位置做处理。
        tmp_child_set_new = []

        if tmp_len > 1:
            for point in tmp_child_set:
                tmp_h, tmp_w = point
                r, g, b = self.img_skel[tmp_h][tmp_w]
                # 前降支
                if r == 0 and g == 255 and b == 0 :
                    # 插入元素到 索引 0 的位置
                    tmp_child_set_new.insert(0, point)
                # 回旋支
                elif r == 0 and g == 0 and b == 255:
                    tmp_child_set_new.append(point)
                    # if len(tmp_child_set_new) > 0:
                    #     tmp_child_set_new.append(point)
                    # else:
                    #     tmp_child_set_new.insert(0, point)


                else:
                    tmp_child_set_new.append(point)

        elif tmp_len == 1:
            tmp_child_set_new = copy.deepcopy(tmp_child_set)

        # 把每个新的邻域添加到本节点的子节点中；
        # for t in tmp_child_set_new:
        for id, t in enumerate(tmp_child_set_new):
            n_h, n_w = t
            tmp_node = node([n_h, n_w])  # 生成一个新节点；
            tt_node, tt_count = self.get_tree_manual_skel_lxh(tmp_node, n_h, n_w)  # 以该节点进行递归；
            ##############################################################
            # 其实是可以直接在这里选择哪些子节点是可以添加 哪些是不可以添加的；
            # if tt_node._all_child_count < self.threshold:
            # if tt_count < self.threshold:
            #     print("当前新节点长度不满足要求，将会被删除: {}".format(tt_count))
            #     # del tmp_node
            #     continue
            ##############################################################
            tmp_node._pre_node = root_node  # 更新当前节点的参数；
            root_node._child.append(tmp_node)
            root_node._child_count += 1
            root_node._all_child_count += tt_count + 1  #

        tmp_len = len(root_node._child)

        if root_node._child_count > 1:  # 对孩子节点处理后再进行判断是否是分叉节点；
            self._branch_nodes.append(root_node)

        '''此部分初步判断不需要，对分叉点进行处理，因为此处只对于 左主干二分叉进行考虑。'''
        # # 判断当前节点是否是分叉点，并根据子分支的孩子节点总数多少来判断是否删除;
        # if tmp_len > 1:  # 这里是只对有多个孩子节点的情况进行处理； #
        #     index_set = []
        #     #
        #     child_dist_set = []  # 存储每个孩子到根节点的距离；
        #     child_count_set = []  #存储每个孩子的总孩子节点数量；
        #     child_index_set = []  # 存储对应孩子节点的索引；
        #     for i in range(0, tmp_len):
        #         t_c = root_node._child[i]
        #         t_h, t_w = root_node._child[i]._value[0:2]
        #         t_dist = np.sqrt((self.all_head_points[0] - t_h)**2 + (self.all_head_points[1] - t_w)**2)
        #         # print("当前分叉点到根节点的距离为: {}".format(t_dist))
        #         # 对每个孩子的分支长度进行判断；
        #         if t_c._all_child_count < self.threshold: #  or t_dist < 5:  # 这里不应当用这样简单的方法进行判断，两个孩子距离都小，但有一个是在长度上满足的；
        #             index_set.append(i)  # 添加不符合长度的分支序号；
        #
        #         if t_dist < 3:  # 分叉点到根节点的距离阈值，小于此阈值，成为无效候选子节点；
        #             child_dist_set.append(t_dist)
        #             child_count_set.append(root_node._child[i]._all_child_count)
        #             child_index_set.append(i)
        #
        #     # 在这里集中对根节点处非法的分叉点进行处理；
        #     if len(child_index_set):
        #         max_index = np.argmax(child_count_set)
        #         print("非法分叉点的最大子节点序号: {}".format(max_index))
        #         print("child_index_set: {}".format(child_index_set))
        #         # child_index_set.remove(max_index)  # 直接保留最大的孩子节点，将其它节点都删除；
        #         child_index_set.pop(max_index)
        #         index_set.extend(child_index_set)
        #
        #     # 这里不应当在for循环以内，否则可能会删除掉所有点；
        #     index_set = list(set(index_set))  # 去掉重复的索引，否则会出错；
        #     nodes_check = self.bi_nodes_check(root_node)  # 对当前的二分叉节点进行检查，排除不合理的二分叉；
        #     if not nodes_check[0] and (nodes_check[1] not in index_set):  # 避免对同一个节点的重复添加；
        #         # print("当前二分叉节点为无效节点，将被删除!")
        #         # tmp_img_for_show = copy.deepcopy(self.img_color)
        #         # cv2.circle(tmp_img_for_show, (c_w, c_h), 10, (0, 255, 0), -1)
        #         # cv2.imshow("bi node check", tmp_img_for_show)
        #         # cv2.waitKey()
        #         index_set.append(nodes_check[1])
        #
        #     # print("以下序号节点将会被删除: {}".format(index_set))
        #     # print("all child count: {}".format(root_node._child_count))
        #     # print("index_set: {}".format(index_set))
        #     if len(index_set):
        #         root_node._child_count -= len(index_set)  # 更新孩子节点数量；
        #         for k in range(len(index_set)):  # 删除长度小于阈值的分支；
        #             root_node._all_child_count -= (
        #                         root_node._child[index_set[k] - k]._all_child_count + 1)  # 更新当前节点后续所有孩子节点的数量；
        #
        #             del root_node._child[index_set[k] - k]  # 删除对应序号的孩子节点；
        #
        # if root_node._child_count > 1:  # 对孩子节点处理后再进行判断是否是分叉节点；
        #     self._branch_nodes.append(root_node)

        return root_node, root_node._all_child_count

    def main(self,
             points,
             img_skel,
             distance,
             img_color,
             POSITION="RIGHT"):
        """
        根据传入的节点，只构建一棵树；

        Args:
            points:  血管树起始点，在骨架上；
            img_skel: 骨架图；
            distance: 原始接口的距离图；

        Returns:
            树的根节点以及所有的分叉节点；
        """

        self.img_record = np.zeros([512, 512], dtype=np.bool)
        self.img_skel = copy.deepcopy(img_skel)
        self.distance = copy.deepcopy(distance)
        self.img_color = img_color  # 这里的img_color是用于对主干中错误存在的二分叉节点进行判断，需要img_color结合体位共同处理；
        self.POSITION = POSITION  # 目的同上；

        self.img_skel_for_show = copy.deepcopy(img_skel)
        # print("img_skel shape: {}".format(img_skel.shape))
        # cv2.imshow("img_skel", img_skel)
        # cv2.waitKey()
        # cv2.imwrite("./dst/img_skel_third.png", img_skel)
        # print("self.img_skel_for_show shape: {}".format(self.img_skel_for_show.shape))
        # self.img_skel_for_show = cv2.cvtColor(self.img_skel_for_show, cv2.COLOR_GRAY2BGR)
        # self.img_skel_for_show = cv2.cvtColor(img_skel, cv2.COLOR_GRAY2BGR)

        # cv2.imshow("img_color_in_tree_building", img_color)
        # cv2.waitKey()

        root_node = node([points[0], points[1]])  # 将参考点坐标封装成树节点类，作为树根节点建树；

        # 建树函数返回的时候表明一颗树建立完成了；
        # 注意清空这里的分叉点记录；
        self._branch_nodes = []
        self.img_record[points[0]][points[1]] = 1  # 这里应当对第一个根节点就进行设置已访问标志；
        # self.count = 0
        begin = datetime.datetime.now()
        self.all_head_points = points
        root_node, root_node_count = self.get_tree(root_node, points[0], points[1])
        # root_node = self.get_tree_without_recursive(root_node)
        end = datetime.datetime.now()
        print("递归建树时间消耗--: {}".format(end - begin))
        # print("self.count: {}".format(self.count))

        return root_node, self._branch_nodes

    def main_manual_skel_lxh(self,
             points,
             img_skel,
             distance = 0,
             img_color = 0,
             POSITION="RIGHT"):
        """
        根据传入的节点，只构建一棵树；

        Args:
            points:  血管树起始点，在骨架上；
            img_skel: 骨架图；
            distance: 原始接口的距离图；

        Returns:
            树的根节点以及所有的分叉节点；
        """

        self.img_record = np.zeros([512, 512], dtype=np.bool)
        self.img_skel = copy.deepcopy(img_skel)
        self.distance = copy.deepcopy(distance)
        self.img_color = img_color  # 这里的img_color是用于对主干中错误存在的二分叉节点进行判断，需要img_color结合体位共同处理；
        self.POSITION = POSITION  # 目的同上；

        self.img_skel_for_show = copy.deepcopy(img_skel)
        # print("img_skel shape: {}".format(img_skel.shape))
        # cv2.imshow("img_skel", img_skel)
        # cv2.waitKey()
        # cv2.imwrite("./dst/img_skel_third.png", img_skel)
        # print("self.img_skel_for_show shape: {}".format(self.img_skel_for_show.shape))
        # self.img_skel_for_show = cv2.cvtColor(self.img_skel_for_show, cv2.COLOR_GRAY2BGR)
        # self.img_skel_for_show = cv2.cvtColor(img_skel, cv2.COLOR_GRAY2BGR)

        # cv2.imshow("img_color_in_tree_building", img_color)
        # cv2.waitKey()

        root_node = node([points[0], points[1]])  # 将参考点坐标封装成树节点类，作为树根节点建树；

        # 建树函数返回的时候表明一颗树建立完成了；
        # 注意清空这里的分叉点记录；
        self._branch_nodes = []
        self.img_record[points[0]][points[1]] = 1  # 这里应当对第一个根节点就进行设置已访问标志；
        # self.count = 0
        begin = datetime.datetime.now()
        self.all_head_points = points
        root_node, root_node_count = self.get_tree_manual_skel_lxh(root_node, points[0], points[1])
        # root_node = self.get_tree_without_recursive(root_node)
        end = datetime.datetime.now()
        print("递归建树时间消耗--: {}".format(end - begin))
        # print("self.count: {}".format(self.count))

        return root_node, self._branch_nodes
