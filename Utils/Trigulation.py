import cv2
import numpy as np

from Utils.AnnotationUtils import annotate_point, annotate_triangle
from Utils.CmmonImageRelateUtils import show_image_with_scale
from Utils.GeometryUtils import get_triangle_bounding_circle, get_convex_point_index, generate_random_point


def is_right(_center, _radius, _point):
    return _point[0] >= _center[0] + _radius


def is_inner(_center, _radius, _point):
    return np.linalg.norm(np.asarray(_point) - _center) < _radius


def add_edge(_edge_buffer, _edge):
    if _edge in _edge_buffer:
        _edge_buffer.remove(_edge)
    elif _edge[::-1] in _edge_buffer:
        _edge_buffer.remove(_edge[::-1])
    else:
        _edge_buffer.append(_edge)


def generate_delaunay_triangles(_all_point_list):
    """
    根据delaunay三角中的空圆准则，生成三角剖分。
    :param _all_point_list: 待处理的点集[[x, y], ...]
    :return: 点集中每个点的连接关系
    """
    delaunay_triangles = []
    triangles_buffer = []
    edge_buffer = []
    all_point_numpy = np.asarray(_all_point_list, dtype=int)
    # 获取点集的凸轮廓点的索引
    convex_point_index = get_convex_point_index(_all_point_list)
    # 对x坐标进行从小到大的排序
    sort_index = np.argsort(all_point_numpy, axis=0)[..., 0].tolist()
    # 预生成一堆三角形
    convex_point_index.append(convex_point_index[0])
    for i, (current_index, next_index) in enumerate(zip(convex_point_index, convex_point_index[1:])):
        edge = [current_index, next_index]
        if convex_point_index[0] not in edge:
            triangles_buffer.append((*edge, convex_point_index[0]))
    # 利用空圆准则判断已有的三角形是否为delaunay三角形
    for point_index in sort_index:
        x, y = all_point_numpy[point_index, ...]
        temp_triangle_num = len(triangles_buffer)
        for _ in range(temp_triangle_num):
            triangle = triangles_buffer.pop(0)
            center, radius = get_triangle_bounding_circle(all_point_numpy[list(triangle)])
            if is_right(center, radius, (x, y)):
                delaunay_triangles.append(triangle)
            elif is_inner(center, radius, (x, y)):
                add_edge(edge_buffer, [triangle[0], triangle[1]])
                add_edge(edge_buffer, [triangle[1], triangle[2]])
                add_edge(edge_buffer, [triangle[2], triangle[0]])
            else:
                triangles_buffer.append(triangle)
        if len(edge_buffer):
            while len(edge_buffer):
                edge = edge_buffer.pop(0)
                if point_index not in edge:
                    triangles_buffer.append((*edge, point_index))
    delaunay_triangles += triangles_buffer
    return delaunay_triangles


if __name__ == '__main__':
    import time

    image_size = (1024, 1024)
    point_image = np.zeros(image_size, dtype=np.uint8)
    random_point_list = generate_random_point(image_size, 5)
    # random_point_list = [(375, 938), (123, 92), (1007, 252), (10, 948), (50, 274)]
    print(random_point_list)
    random_point_image = annotate_point(point_image, random_point_list)
    start_time = time.time()
    result = generate_delaunay_triangles(random_point_list)
    print(time.time() - start_time, ': ', len(result))
    for m_triangle_index in result:
        m_triangle = np.asarray(random_point_list)[list(m_triangle_index)]
        random_point_image = annotate_triangle(random_point_image, m_triangle)
        show_image_with_scale('random_point_image', random_point_image)
        cv2.waitKey(200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
