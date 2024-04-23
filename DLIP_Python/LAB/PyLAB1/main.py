"""
========================================================================================================================
                            DLIP LAB 3 Python Tension Detection of Rolling Metal Sheet
                                          Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.23
@Version: 0.0.0
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from utility.utils import *

def main():
    src = cv.imread("../../Image/Simple_Dataset/LV3_simple.png")
    pre_src = Preprocess(src)
    cv.line(src, (0,src.shape[0]-120), (src.shape[1],src.shape[0]-120), (255,0,255), 2, cv.LINE_AA)
    cv.line(src, (0,src.shape[0]-250), (src.shape[1],src.shape[0]-250), (255,255,0), 2 ,cv.LINE_AA)
    showImage("LV1", src)
    showImage("LV1_Out", pre_src)
    cv.waitKey(0)


"""
img: BGR image
return: Grayscale image after preprocessing (Filtering, Equalize Histogram ...) 
"""
def Preprocess(img: np.ndarray) -> np.ndarray:
    _dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(0,15):
        _dst = cv.medianBlur(_dst,5)
    cv.equalizeHist(_dst, _dst)
    _dst = cv.Canny(_dst,40,255)
    return _dst

if __name__ == '__main__':
    print(__doc__)
    main()