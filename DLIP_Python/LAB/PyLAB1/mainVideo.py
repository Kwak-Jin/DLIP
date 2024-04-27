"""
========================================================================================================================
                            DLIP LAB 3 Python Tension Detection of Rolling Metal Sheet
                                              Challenging Video
                                          Handong Global University
========================================================================================================================
@Author  : Jin Kwak
@Created : 2024.04.27
@Modified:2024.04.28
@Version : 0.0.2
"""
import cv2 as cv
import numpy as np

N_MedianIter = 20
kernel = 5
YOK = 1
NOK = 0
level2 = 250
level3 = 120
""" ROI length and width"""
ROI_y = 200
ROI_h = 1080 -ROI_y
ROI_x = 200
ROI_w = 700


def main():
    cap = cv.VideoCapture('../../Image/LAB3_Video.mp4')
    if not cap.isOpened():
        print('Cannot open Video Capture')

    discontinuity = 20
    kernelSize = 3
    #Threshold Values for height


    cv.namedWindow('MyVideo', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Src', cv.WINDOW_AUTOSIZE)

    while True:
        # Read a new frame from video
        ret, frame = cap.read()
        if not checkEnd(ret):
            break

        roi = frame[ROI_y:ROI_y + ROI_h, ROI_x:ROI_x + ROI_w]
        psrc = roi.copy()
        pre_src = Preprocess(roi)

        fx = np.zeros((ROI_w), np.int16)
        xx = np.zeros((ROI_w), np.int16)
        newFx = np.zeros((ROI_w), np.int32)
        totalX = np.zeros((ROI_w), np.int16)
        for idx in range(0, ROI_w):  # col
            cutIdx = 0
            for tdx in range(ROI_h - 1, 0, -1):  # row
                if pre_src[tdx, idx] == 255:
                    fx[idx] = tdx  # row
                    frame = cv.circle(frame, (ROI_x + idx, ROI_y+fx[idx]), 1, (255, 0, 255), -1)
                    psrc = cv.circle(psrc, (idx, fx[idx]), 1, (255, 0, 255), -1)
                    break
            xx[idx] = idx
            if (idx > 0) and (np.abs(fx[idx] - fx[idx - 1]) >= discontinuity):
                cutIdx = idx
                break

        fx = fx[:cutIdx]
        xx = xx[:cutIdx]
        for idx in range(0, ROI_w):
            totalX[idx] = idx
        coeff = np.polyfit(xx, fx, 2)
        if coeff[0] > 0:
            coeff[0]  = -coeff[0]

        for cnt in range(0, ROI_w):
            newFx[cnt] = (int)(coeff[0] * (totalX[cnt] ** 2) + coeff[1] * totalX[cnt] + coeff[2])
        for idx in range(0, ROI_w):
            psrc = cv.circle(psrc, (idx, newFx[idx]), 1, (255, 255, 0), -1)

        maxBend = max(newFx)
        if maxBend < frame.shape[0] - level3 and maxBend >= frame.shape[0] - level2:
            print("Level 2 detected")
            _str = "Level 2 detected"
        elif maxBend > frame.shape[0] - level3:
            print("Level 3 Detected")
            _str = ("Level 3 Detected")
        else:
            print("Level 1 Detected")
            _str = ("Level 1 Detected")

        cv.putText(frame, _str, ((int)(ROI_h/2),(int)(ROI_w/2)) , cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Src',frame)
        cv.imshow('Canny',pre_src)
        cv.imshow("CurveFit", psrc)


    cv.destroyAllWindows()
    cap.release()

def Preprocess(img: np.ndarray) -> np.ndarray:
    _dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(0, N_MedianIter):
        _dst = cv.medianBlur(_dst, kernel)
    for i in range(0, 2):
        _dst = cv.GaussianBlur(_dst, (kernel,kernel), 0)

    cv.equalizeHist(_dst, _dst)
    _dst = cv.Canny(_dst, 40, 255)
    return _dst

def checkEnd(ret:bool) -> int:
    if cv.waitKey(30) & 0xFF == 27 or not ret:
        print('Press ESC to stop')
        flag = NOK
    else:
        flag = YOK
    return flag

def drawLine(img: np.ndarray) -> np.ndarray:
    cv.line(img, (0, img.shape[0] - level3), (img.shape[1], img.shape[0] - level3), (255, 0, 255), 2,
            cv.LINE_AA)  # Level 3
    cv.line(img, (0, img.shape[0] - level2), (img.shape[1], img.shape[0] - level2), (255, 255, 0), 2,
            cv.LINE_AA)  # Level 2
    return img

if __name__ == '__main__':
    main()