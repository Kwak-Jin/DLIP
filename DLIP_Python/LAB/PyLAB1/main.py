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

def main():
    src = cv.imread("../../Image/Simple_Dataset/LV1_simple.png")
    pre_src = Preprocess(src)
    edge    = cv.Canny(pre_src,40,255)
    cv.imshow("LV1", src)
    cv.imshow("LV1_Out", edge)

    cv.waitKey(0)

def Preprocess(img: np.ndarray) -> np.ndarray:
    _dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(0,15):
        _dst = cv.medianBlur(_dst,5)
    cv.equalizeHist(_dst, _dst)
    return _dst

if __name__ == '__main__':
    print(__doc__)
    main()