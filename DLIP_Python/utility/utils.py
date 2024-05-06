"""
========================================================================================================================
                                             DLIP Python library
                                          Handong Global University
========================================================================================================================
@Author  : Jin Kwak
@Created : 2024.04.23
@Modified: 2024.-
@Version : 0.0.0
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

"""
filename: filename of image (Figure title)
img1: image
"""
def showImage(filename:str, imag1: np.ndarray) -> None:
    cv.namedWindow(filename, cv.WINDOW_GUI_NORMAL)
    cv.imshow(filename, imag1)

def histogram_analysis(imag1: np.ndarray) -> None:
    hist_full = cv.calcHist([imag1],[0],None,[256],[0,256])
    plt.plot(hist_full)
    plt.xlim([0,256])
    plt.savefig('histogram.png')
    plt.show()