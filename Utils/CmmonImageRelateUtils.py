import cv2


def show_image_with_scale(_window_name, _image, _scale=0.5):
    cv2.imshow(_window_name, cv2.resize(_image, (0, 0), fx=_scale, fy=_scale))
