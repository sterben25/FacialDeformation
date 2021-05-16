import cv2


def annotate_point(_annotate_image, _all_point_list, _radius=5):
    annotate_image = _annotate_image.copy()
    for (x, y) in _all_point_list:
        cv2.circle(annotate_image, (x, y), radius=_radius, color=255, thickness=-1)
    return annotate_image


def annotate_triangle(_annotate_image, _triangle, _color=255, _thickness=2):
    to_return_image = _annotate_image.copy()
    cv2.line(to_return_image, tuple(_triangle[0]), tuple(_triangle[1]), color=_color, thickness=_thickness)
    cv2.line(to_return_image, tuple(_triangle[1]), tuple(_triangle[2]), color=_color, thickness=_thickness)
    cv2.line(to_return_image, tuple(_triangle[2]), tuple(_triangle[0]), color=_color, thickness=_thickness)
    return to_return_image
