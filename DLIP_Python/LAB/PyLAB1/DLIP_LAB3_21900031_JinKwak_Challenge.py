"""
========================================================================================================================
                            DLIP LAB 3 Python Tension Detection of Rolling Metal Sheet
                                            Challenging Dataset
                                          Handong Global University
========================================================================================================================
@Author  : Jin Kwak
@Created : 2024.04.23
@Modified: 2024.05.07
@Version : 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

N_MedianIter = 25
kernel = 5
YOK = 1
NOK = 0
level2 = 250
level3 = 120
""" ROI length and width"""
ROI_y = 200
ROI_h = 1080 - ROI_y
ROI_x = 125
ROI_w = 575
def main():
    src = cv.imread("../../Image/Challenging_Dataset/LV3.png")
    discontinuity = 20
    roi = src[ROI_y:ROI_y + ROI_h, ROI_x:ROI_x + ROI_w]
    psrc = roi.copy()
    pre_src = Preprocess(psrc)
    cv.line(src, (0,src.shape[0]-120), (src.shape[1],src.shape[0]-120), (255,0,255), 2, cv.LINE_AA) # Level 3
    cv.line(src, (0,src.shape[0]-250), (src.shape[1],src.shape[0]-250), (255,255,0), 2 ,cv.LINE_AA) # Level 2
    fx1 = []
    fx = []
    xx1 = []
    xx = []
    totalX = np.zeros((ROI_w - 1), np.int16)
    for idx in range(0, ROI_w):  # col
        for tdx in range(ROI_h - 1, 0, -1):  # row
            if pre_src[tdx, idx] == 255:
                fx.append(tdx)  # level
                xx.append(idx)
                if len(xx) > 1:
                    psrc = cv.circle(psrc, (xx[-1], fx[-1]), 1, (255, 0, 255), -1)
                break
        if (len(fx) > 2) and ((np.abs(fx[-1] - fx[-2]) >= discontinuity) or (np.abs(xx[-1] - xx[-2]) >= 20)):
            break
    for idx in range(ROI_w - 1, 0, -1):  # col
        for tdx in range(0, ROI_h - 1):  # row
            if pre_src[tdx, idx] == 255:
                fx1.append(tdx)  # row
                xx1.append(idx)
                if len(xx1) > 1:
                    psrc = cv.circle(psrc, (xx1[-1], fx1[-1]), 1, (255, 0, 255), -1)
                break
        if (idx < ROI_w - 2) and (np.abs(fx1[-1] - fx1[-2]) >= discontinuity):
            break

    xx = np.hstack([xx, xx1])
    fx = np.hstack([fx, fx1])
    for idx in range(0, ROI_w - 1):
        totalX[idx] = idx
    coeff = np.polyfit(xx, fx, 2)
    if (coeff[0]>0):
        coeff[0] = -coeff[0]
    p = np.poly1d(coeff)
    # Predicted values
    y_pred = p(xx)

    rmse = np.sqrt(np.mean((fx - y_pred) ** 2))

    maxY = 0
    for x in range(0, src.shape[1] - 1):
        newY = coeff[0] * (x - ROI_x) ** 2 + coeff[1] * (x - ROI_x) + coeff[2] + ROI_y
        if newY > maxY:
            maxY = newY
        newY = int(newY)
        src = cv.circle(src, (x, newY), 1, (255, 255, 0), -1)
    """ After Polyfit  """
    if maxY < src.shape[0] - level3 and maxY >= src.shape[0] - level2:
        _str = "Level: 2"
    elif maxY > src.shape[0] - level3:
        _str = "Level: 3"
    else:
        _str = "Level: 1"

    # src = drawLevelBound(src.shape[0] - src)
    src = drawLevelBound(src)
    bend = 'Score:' + str(src.shape[0]-maxY)
    __str = "RMSE :" + str(rmse)
    cv.putText(src, _str, ((int)(ROI_w) + 400, (int)(ROI_h / 2) - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(src, bend, ((int)(ROI_w) + 400, (int)(ROI_h / 2)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv.putText(src, __str, ((int)(ROI_w)+400, (int)(ROI_h / 2)+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow('Src', src)
    cv.imshow('Canny', pre_src)
    cv.imshow("CurveFit", psrc)
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

def drawLevelBound(img: np.ndarray) -> np.ndarray:
    cv.line(img, (0, img.shape[0] - level3), (img.shape[1], img.shape[0] - level3), (255, 0, 255), 2,
            cv.LINE_AA)  # Level 3
    cv.line(img, (0, img.shape[0] - level2), (img.shape[1], img.shape[0] - level2), (255, 255, 0), 2,
            cv.LINE_AA)  # Level 2
    return img

if __name__ == '__main__':
    print(__doc__)
    main()