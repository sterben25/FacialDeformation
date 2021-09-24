import cv2
import numpy as np


def show_image_with_scale(_window_name, _image, _scale=0.5):
    cv2.imshow(_window_name, cv2.resize(_image, (0, 0), fx=_scale, fy=_scale))


def interpolation(_src_coordinates, _dst_coordinates, _src_image, _dst_image, _mode='bilinear'):
    support_interpolation = ['bilinear', 'nearest']
    if _mode not in support_interpolation:
        raise NotImplementedError(f'为定义的插值方法：{_mode}, 目前支持的方法有：{support_interpolation}')
    if _mode == 'bilinear':
        return bilinear_interpolation(_src_coordinates, _dst_coordinates, _src_image, _dst_image)
    elif _mode == 'nearest':
        return nearest_interpolation(_src_coordinates, _dst_coordinates, _src_image, _dst_image)


def nearest_interpolation(_src_coordinates, _dst_coordinates, _src_image, _dst_image):
    src_h, src_w = _src_image.shape[:2]
    dst_h, dst_w = _dst_image.shape[:2]
    src_x = np.clip(_src_coordinates[:, 0].astype(int).flatten(), 0, src_w - 1)
    src_y = np.clip(_src_coordinates[:, 1].astype(int).flatten(), 0, src_h - 1)
    dst_x = np.clip(_dst_coordinates[:, 0].astype(int).flatten(), 0, dst_w - 1)
    dst_y = np.clip(_dst_coordinates[:, 1].astype(int).flatten(), 0, dst_h - 1)
    _dst_image[dst_y, dst_x, :] = _src_image[src_y, src_x, :]
    return _dst_image


def bilinear_interpolation(_src_coordinates, _dst_coordinates, _src_image, _dst_image):
    """
    矩阵运算加速的双线性插值
    :param _src_coordinates: 原始图像上的坐标
    :param _dst_coordinates: 插值之后的图像坐标
    :param _src_image: 原始图像
    :param _dst_image: 插值之后的图像
    :return:
    """
    src_h, src_w = _src_image.shape[:2]
    dst_h, dst_w = _dst_image.shape[:2]
    # 求最临近的四个点的坐标
    src_floor_x, src_ceil_x = np.floor(_src_coordinates[:, 0]), np.ceil(_src_coordinates[:, 0])
    src_floor_y, src_ceil_y = np.floor(_src_coordinates[:, 1]), np.ceil(_src_coordinates[:, 1])
    src_x, src_y = _src_coordinates[:, 0], _src_coordinates[:, 1]
    # 计算最临近的四个点的权重
    top_left_scale = np.repeat(((src_ceil_x - src_x) * (src_ceil_y - src_y))[..., None], 3, axis=1)
    top_right_scale = np.repeat(((src_x - src_floor_x) * (src_ceil_y - src_y))[..., None], 3, axis=1)
    bottom_left_scale = np.repeat(((src_ceil_x - src_x) * (src_y - src_floor_y))[..., None], 3, axis=1)
    bottom_right_scale = np.repeat(((src_x - src_floor_x) * (src_y - src_floor_y))[..., None], 3, axis=1)
    # 防止值越界
    dst_x = np.clip(_dst_coordinates[:, 0].flatten(), 0, dst_w - 1).astype(int)
    dst_y = np.clip(_dst_coordinates[:, 1].flatten(), 0, dst_h - 1).astype(int)
    src_floor_x = np.clip(src_floor_x.flatten(), 0, src_w - 1).astype(int)
    src_ceil_x = np.clip(src_ceil_x.flatten(), 0, src_w - 1).astype(int)
    src_floor_y = np.clip(src_floor_y.flatten(), 0, src_h - 1).astype(int)
    src_ceil_y = np.clip(src_ceil_y.flatten(), 0, src_h - 1).astype(int)
    # 根据权重计算新的像素值
    _dst_image[dst_y, dst_x, :] = _src_image[src_floor_y, src_floor_x, :] * top_left_scale + \
                                  _src_image[src_floor_y, src_ceil_x, :] * top_right_scale + \
                                  _src_image[src_ceil_y, src_ceil_x, :] * bottom_right_scale + \
                                  _src_image[src_ceil_y, src_floor_x, :] * bottom_left_scale
    return _dst_image
