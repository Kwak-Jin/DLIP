"""
========================================================================================================================
                            DLIP LAB 3 Python Tension Detection of Rolling Metal Sheet
                                            Challenging Images
                                          Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.26
@Version: 0.0.1
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from utility.utils import *

def main():
    src = cv.imread("../../Image/Challenging_Dataset/LV1.png")
    """ ROI length and width"""
    y= 0
    h= 1040
    x= 50
    w= 700
    discontinuity = 20
    roi = src[y:y + h, x:x + w]
    psrc = roi.copy()
    pre_src = Preprocess(psrc)
    cv.line(src, (0,src.shape[0]-120), (src.shape[1],src.shape[0]-120), (255,0,255), 2, cv.LINE_AA) # Level 3
    cv.line(src, (0,src.shape[0]-250), (src.shape[1],src.shape[0]-250), (255,255,0), 2 ,cv.LINE_AA) # Level 2
    fx = np.zeros((w),np.int16)
    xx = np.zeros((w), np.int16)
    totalX= np.zeros((w),np.int16)

    showImage("LV1", src)
    showImage("LV1_Out", pre_src)
    showImage("ROI",psrc)
    cv.waitKey(0)

"""
img: BGR image
return: Grayscale image after preprocessing (Filtering, Equalize Histogram ...) 
"""
def Preprocess(img: np.ndarray) -> np.ndarray:
    _dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(0,50):
        _dst = cv.medianBlur(_dst,5)
    for i in range(0,3):
        _dst = cv.GaussianBlur(_dst, (5, 5), 0)

    cv.equalizeHist(_dst, _dst)
    _dst = cv.Canny(_dst,40,255)
    return _dst

if __name__ == '__main__':
    print(__doc__)
    main()