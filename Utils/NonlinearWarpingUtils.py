import numpy as np

"""
非线性变换相关的算法
"""


def local_translation_warp(_image, _start_coordinate, _end_coordinate, _radius, _reverse=True,
                           _interpolation='bilinear'):
    """
    局部平移(液化形变)的公式：
        u = x - ((r^2 - |x - c|^2) / (r^2 - |x - c|^2 + |m - c|^2))^2 * (m - c)
    其中x为平移后的坐标，u为平移前的坐标，r为平移半径，c为起始点，m为终点。
    :param _image:  待处理的图片
    :param _start_coordinate: 局部平移的起始点， tuple: (start_x, start_y)
    :param _end_coordinate:  局部平移的终点，tuple: (end_x, end_y)
    :param _radius: 局部平移的半径，以起始点为圆心，在半径范围内的点都需要进行处理。
    :param _reverse: 为False时平移的方向为终点到起始点，为True时，平移的方向为起始点到终点。
    :param _interpolation: 计算平移后的像素值的方法，默认为双线性插值。
    :return:
    """
    _h, _w = _image.shape[:2]
    start_x, start_y = _start_coordinate
    end_x, end_y = _end_coordinate
    radius_2 = float(_radius * _radius)
    m_c_2 = np.sum(np.square(np.asarray(_start_coordinate) - np.asarray(_end_coordinate)))
    warped_image = _image.copy()
    warp_rows = np.linspace(0, _h - 1, num=_h, dtype=int)
    warp_cols = np.linspace(0, _w - 1, num=_w, dtype=int)
    warp_cols, warp_rows = np.meshgrid(warp_cols, warp_rows)
    distance = np.square(warp_cols - start_x) + np.square(warp_rows - start_y)
    ratio = np.square((radius_2 - distance) / (radius_2 - distance + m_c_2))
    if _reverse:
        original_cols = np.clip(warp_cols + ratio * (end_x - start_x), 0, _w - 1)
        original_rows = np.clip(warp_rows + ratio * (end_y - start_y), 0, _h - 1)
    else:
        original_cols = np.clip(warp_cols - ratio * (end_x - start_x), 0, _w - 1)
        original_rows = np.clip(warp_rows - ratio * (end_y - start_y), 0, _h - 1)
    if _interpolation == 'bilinear':
        warp_cols = warp_cols.flatten()
        warp_rows = warp_rows.flatten()
        original_cols = original_cols.flatten()
        original_rows = original_rows.flatten()
        left_original_cols = original_cols.astype(int)
        left_original_rows = original_rows.astype(int)
        right_original_cols = np.clip((original_cols + 1).astype(int), 0, _w - 1)
        right_original_rows = np.clip((original_rows + 1).astype(int), 0, _h - 1)
        right_rows_diff = (right_original_rows - original_rows)[:, None].repeat(3, axis=1)
        right_cols_diff = (right_original_cols - original_cols)[:, None].repeat(3, axis=1)
        left_rows_diff = (original_rows - left_original_rows)[:, None].repeat(3, axis=1)
        left_cols_diff = (original_cols - left_original_cols)[:, None].repeat(3, axis=1)
        part1 = _image[left_original_rows, left_original_cols] * right_cols_diff * right_rows_diff
        part2 = _image[left_original_rows, right_original_cols] * left_cols_diff * right_rows_diff
        part3 = _image[right_original_rows, left_original_cols] * right_cols_diff * left_rows_diff
        part4 = _image[right_original_rows, right_original_cols] * left_cols_diff * left_rows_diff
        warped_image[warp_rows, warp_cols] = part1 + part2 + part3 + part4
    else:
        filtered_image = _image.copy()
        warp_cols = warp_cols.flatten()
        warp_rows = warp_rows.flatten()
        original_cols = original_cols.flatten().astype(int)
        original_rows = original_rows.flatten().astype(int)
        warped_image[warp_rows, warp_cols] = filtered_image[original_rows, original_cols]
    warp_mask = distance < radius_2
    warp_mask = np.stack([warp_mask, warp_mask, warp_mask], axis=2)
    warped_image = warped_image * warp_mask + _image * (1 - warp_mask)
    return warped_image.astype(np.uint8)


if __name__ == '__main__':
    pass
