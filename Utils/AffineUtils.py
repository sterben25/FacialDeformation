import numpy as np


def get_affine_matrix(_tx=0, _ty=0, _sx=1., _sy=1., _rotation=0., _cx=0, _cy=0):
    """
    根据给定的参数生成防射变换矩阵
    :param _tx: x的位移量
    :param _ty: y的位移量
    :param _sx: x的缩放尺度
    :param _sy: y的缩放尺度
    :param _rotation: 图像旋转角度（顺时针）
    :param _cx: 旋转x轴中心
    :param _cy: 旋转y轴中心
    :return: 仿射变换矩阵
    """
    to_return_affine_matrix = np.eye(3)
    # 平移
    translation = to_return_affine_matrix.copy()
    translation[:, 2] = np.asarray(
        [_tx, _ty, 1]
    )
    # 缩放
    scaling = to_return_affine_matrix.copy()
    scaling[0:2, 0:2] = np.asarray([
        [_sx, 0],
        [0, _sy],
    ])
    # 旋转
    # Opencv坐标系中的坐标远点在左上角，因此负号在[0, 1]位置时为顺时针旋转，在[1, 0]位置时为逆时针旋转
    _rotation = _rotation * np.pi / 180
    rotation = to_return_affine_matrix.copy()
    rotation[0:2, 0:2] = np.asarray([
        [np.cos(_rotation), -np.sin(_rotation)],
        [np.sin(_rotation), np.cos(_rotation)],
    ])
    # 设置旋转中心
    rotate_x, rotate_y = np.dot(np.asarray([_cx, _cy, 1]), np.transpose(rotation))[:2]
    translate_x, translate_y = rotate_x - _cx, rotate_y - _cy
    rotate_translation = to_return_affine_matrix.copy()
    rotate_translation[:, 2] = np.asarray(
        [translate_x, translate_y, 1]
    )
    for m_affine_matrix in [translation, scaling, rotation, rotate_translation]:
        to_return_affine_matrix = np.dot(np.transpose(to_return_affine_matrix), m_affine_matrix)
    return to_return_affine_matrix


def affine_transformation_fast(_affine_matrix, _image, _keep_size=True):
    """
    用矩阵运算进行加速的仿射变换
    :param _affine_matrix:  仿射变换矩阵
    :param _image:  待处理的图像
    :param _keep_size:  是否保持原图像的大小
    :return:  变换后的图像
    """
    h, w = _image.shape[:2]
    if _keep_size:
        # 缩放
        scale_x, scale_y = 1, 1
        # 平移
        translation_x, translation_y = 0, 0
        # 缩放
    else:
        # 缩放
        scale_x = np.sqrt(np.sum(_affine_matrix[0, 0:2] ** 2))
        scale_y = np.sqrt(np.sum(_affine_matrix[1, 0:2] ** 2))
        # 平移
        translation_x, translation_y = (_affine_matrix[0, 2]), _affine_matrix[1, 2]
        # 缩放
    affine_h, affine_w = int(h * scale_y + abs(translation_x)), int(w * scale_x + abs(translation_y))
    to_return_image = np.zeros([affine_h, affine_w, 3], dtype=np.uint8)
    x_matrix = np.arange(0, w).reshape(1, w).repeat(h, axis=0)
    y_matrix = np.arange(0, h).reshape(h, 1).repeat(w, axis=1)
    coordinate_matrix = np.stack([x_matrix, y_matrix, np.ones_like(x_matrix)], axis=0)
    coordinate_matrix = np.reshape(coordinate_matrix, (3, -1))
    affine_coordinate_matrix = np.dot(_affine_matrix, coordinate_matrix).astype(int)
    affine_coordinate_matrix = np.reshape(affine_coordinate_matrix, (3, h, w))
    affine_x = affine_coordinate_matrix[0, ...]
    affine_y = affine_coordinate_matrix[1, ...]
    x, y = x_matrix.flatten(), y_matrix.flatten()
    affine_x, affine_y = np.clip(affine_x.flatten(), 0, affine_w - 1), np.clip(affine_y.flatten(), 0, affine_h - 1)
    to_return_image[affine_y, affine_x] = _image[y, x]
    return to_return_image


def affine_transformation(_affine_matrix, _image):
    """
    仿射变换
    :param _affine_matrix:  仿射变换矩阵
    :param _image:  待处理的图像
    :return:  变换后的图像
    """
    h, w = _image.shape[:2]
    to_return_image = np.zeros_like(_image)
    for x in range(w):
        for y in range(h):
            coordinate = np.asarray([x, y, 1]).reshape(3, 1)
            _x, _y, _ = np.dot(_affine_matrix, coordinate).astype(int)
            if 0 <= _x < w and 0 <= _y < h:
                to_return_image[_y, _x, ...] = _image[y, x, ...]
    return to_return_image


if __name__ == '__main__':
    import sys
    import cv2
    import time

    assert len(sys.argv) == 2, '使用方法python AffineUtils.py 图像路径'
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (0, 0), fx=1, fy=1)
    h, w = image.shape[:2]
    start_time = time.time()
    affine_matrix = get_affine_matrix(_rotation=15, _cx=w/3, _cy=h/2)
    affine_image = affine_transformation_fast(affine_matrix, image, _keep_size=True)
    print('fast: ', time.time() - start_time)
    # start_time = time.time()
    # affine_image = affine_transformation(affine_matrix, image)
    # print('normal', time.time() - start_time)
    cv2.imshow('image', cv2.resize(image, (0, 0), fx=1, fy=1))
    cv2.imshow('affine_image', cv2.resize(affine_image, (0, 0), fx=1, fy=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
