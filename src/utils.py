"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import cv2
import numpy as np


def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)


__all__  = ["pre_processing"]