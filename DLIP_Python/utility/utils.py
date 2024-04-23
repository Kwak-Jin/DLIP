"""
========================================================================================================================
                                             DLIP Python library
                                          Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.23
@Version: 0.0.0
"""

import numpy as np
import cv2 as cv

"""
filename: filename of image (Figure title)
img1: image
"""
def showImage(filename:str, imag1: np.ndarray) -> None:
    cv.namedWindow(filename, cv.WINDOW_GUI_NORMAL)
    cv.imshow(filename, imag1)