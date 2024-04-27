"""
========================================================================================================================
                            DLIP LAB 3 Python Tension Detection of Rolling Metal Sheet
                                                Simple Dataset
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
    src = cv.imread("../../Image/Challenging_Dataset/LV2.png")
    """ ROI length and width"""
    ROI_y= 0
    ROI_h= 1040
    ROI_x= 50
    ROI_w= 700
    level2 = 250
    level3 = 120
    discontinuity = 20
    roi = src[ROI_y:ROI_y + ROI_h, ROI_x:ROI_x + ROI_w]
    psrc = roi.copy()
    pre_src = Preprocess(psrc)
    cv.line(src, (0,src.shape[0]-120), (src.shape[1],src.shape[0]-120), (255,0,255), 2, cv.LINE_AA) # Level 3
    cv.line(src, (0,src.shape[0]-250), (src.shape[1],src.shape[0]-250), (255,255,0), 2 ,cv.LINE_AA) # Level 2
    fx = np.zeros((ROI_w),np.int16)
    xx = np.zeros((ROI_w), np.int16)
    newFx = np.zeros((ROI_w), np.float64)
    totalX= np.zeros((ROI_w),np.int16)
    for idx in range(0, ROI_w):    # col
        for tdx in range(ROI_h-1,0,-1): # row
            if pre_src[tdx,idx] == 255:
                fx[idx] = tdx  # row
                src = cv.circle(src, (ROI_x+idx,fx[idx]), 1, (255,0,255), -1)
                psrc = cv.circle(psrc, (idx,fx[idx]), 1, (255,0,255), -1)
                break
        xx[idx] = idx
        if (idx> 0)and (np.abs(fx[idx] -fx[idx-1])>= discontinuity):
            cutIdx = idx
            break
    fx = fx[:cutIdx]
    xx = xx[:cutIdx]
    for idx in range(0, ROI_w):
        totalX[idx] = idx
    coeff = np.polyfit(xx,fx,2)

    for cnt in range(0, ROI_w):
        newFx[cnt] = coeff[0]*(totalX[cnt]**2) + coeff[1]*totalX[cnt] + coeff[2]
    newFx = newFx.astype(np.int32)
    for idx in range(0,ROI_w):
        psrc = cv.circle(psrc, (idx,newFx[idx]),1,(255,255,0), -1)

    maxBend = max(newFx)
    if maxBend < src.shape[0]-level3 and maxBend >= src.shape[0]-level2:
        print("Level 2 detected")
    elif maxBend > src.shape[0]-level3:
        print("Level 3 Detected")
    else:
        print("Level 1 Detected")

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
    for i in range(0,20):
        _dst = cv.medianBlur(_dst,5)
    for i in range(0,3):
        _dst = cv.GaussianBlur(_dst, (5, 5), 0)

    cv.equalizeHist(_dst, _dst)
    _dst = cv.Canny(_dst,40,255)
    return _dst

if __name__ == '__main__':
    print(__doc__)
    main()