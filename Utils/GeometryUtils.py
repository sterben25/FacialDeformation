import cv2
import numpy as np


def generate_random_point(_size, _num=3):
    """
    在制定范围内随机生成指定个数的点
    :param _size: 宽和高的范围，可以理解为画布的大小[w, h]
    :param _num: 点的个数
    :return: [[x, y],...]
    """
    to_return_points = []
    max_x, max_y = _size
    counter = 0
    while counter < _num:
        random_x = np.random.randint(max_x)
        random_y = np.random.randint(max_y)
        if (random_x, random_y) not in to_return_points:
            to_return_points.append((random_x, random_y))
            counter += 1
    return to_return_points


def get_bounding_triangle(_rect_info):
    """
    获取给定矩形的最小外接三角形
    :param _rect_info: [min_x, min_y, w, h]
    :return:
    """
    min_x, min_y, w, h = _rect_info
    max_x, max_y = min_x + w, min_y + h
    expand_w, expand_h = int(h * (np.sqrt(3) / 3)), int(w * (np.sqrt(3) / 2))
    top_point = (int((min_x + max_x) / 2), min_y - expand_h)
    left_point = (min_x - expand_w, max_y)
    right_point = (max_x + expand_w, max_y)
    bounding_rectangle = [top_point, left_point, right_point]
    expand_w = max(expand_w, abs(min_x - expand_w))
    expand_h = max(expand_w, abs(min_y - expand_h))
    return bounding_rectangle, expand_w, expand_h


def get_bounding_rect(_all_point, _expand_ratio=1.):
    """
    获取给定点集的最小外接矩形，并将该矩形进行扩大
    :param _all_point: 给定的点集[[x, y],...]
    :param _expand_ratio: 矩形扩大的倍率
    :return: min_x, min_y, w, h
    """
    all_point = np.asarray(_all_point)
    min_x, min_y = np.min(all_point[..., 0]), np.min(all_point[..., 1])
    max_x, max_y = np.max(all_point[..., 0]), np.max(all_point[..., 1])
    w, h = max_x - min_x, max_y - min_y
    expand_x = ((_expand_ratio - 1) * w) / 2
    expand_y = ((_expand_ratio - 1) * h) / 2
    min_x, min_y = int(min_x - expand_x), int(min_y - expand_y)
    w, h = int(w * _expand_ratio), int(h * _expand_ratio)
    return min_x, min_y, w, h


def get_triangle_bounding_circle(_triangle):
    """
    获取给定点三角形的外接圆。
    :param _triangle: 3行2列的二维数组，存储着一个三角形的三个顶点
    :return: 圆心坐标以及半径大小
    """
    rows, cols = _triangle.shape

    A = np.bmat([[2 * np.dot(_triangle, _triangle.T), np.ones((rows, 1))],
                 [np.ones((1, rows)), np.zeros((1, 1))]])
    # np.bmat从数组建立矩阵, pts.T是转置

    b = np.hstack((np.sum(_triangle * _triangle, axis=1), np.ones((1))))
    # hstack的字母h来自于horizontal，表示两个数组是水平的，hstack((a,b))将把b排在a的右边的意思

    x = np.linalg.solve(A, b)
    # solve函数有两个参数a和b：a是一个N*N的二维数组，而b是一个长度为N的一维数组；
    # solve函数找到一个长度为N的一维数组x，使得a和x的矩阵乘积正好等于b，数组x就是多元一次方程组的解

    bary_coords = x[:-1]  # 除去x数组的最后一个元素

    # tile函数将一个数组重复一定次数形成一个新的数组
    # tile(a,(m,n)):即是把a数组里面的元素复制n次放进一个数组c中，然后再把数组c复制m次放进数组b
    # np.sum(arr, axis=0), 表示按列相加 (axis=1表示按行相加)
    center = np.sum(_triangle * np.tile(bary_coords.reshape((_triangle.shape[0], 1)), (1, _triangle.shape[1])), axis=0)
    center = np.asarray(center, dtype=int)
    radius = int(np.linalg.norm(center - _triangle[0, ...]))
    return center, radius


def get_convex_point_index(_all_point_list):
    """
    获取给定点集的最小凸轮廓
    :param _all_point_list: [[x, y],...]
    :return:
    """
    to_return_convex_point_index = []
    all_point_numpy = np.asarray(_all_point_list)
    convex_points = cv2.convexHull(all_point_numpy)
    for convex_point in convex_points:
        point_index = _all_point_list.index(tuple(convex_point[0, :]))
        to_return_convex_point_index.append(point_index)
    return to_return_convex_point_index
