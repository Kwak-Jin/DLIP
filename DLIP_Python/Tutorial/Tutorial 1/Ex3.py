"""
========================================================================================================================
                                        DLIP Exercise 3 (Tutorial 1) Python OpenCV
                                               Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.23
"""
import cv2 as cv
import numpy as np


def main():
    src = cv.imread('coin.jpg')
    dst = src.copy()
    dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    for i in range(0,15):
        dst = cv.medianBlur(dst,5)
    ret, dst_thr = cv.threshold(dst,112,255,cv.THRESH_BINARY)
    for i in range(0,10):
        dst_thr = cv.morphologyEx(dst_thr,cv.MORPH_CLOSE,np.ones((5,5),np.uint8))
    rows = dst_thr.shape[0]

    circles = cv.HoughCircles(dst_thr, cv.HOUGH_GRADIENT, 1, rows / 20,
                              param1=120, param2=10,
                              minRadius=25, maxRadius=100)
    fivehundred = 0
    hundred     = 0
    fifty    = 0
    ten     = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx_circle in circles[0, :]:
            center = (idx_circle[0], idx_circle[1])
            radius = idx_circle[2]
            if(radius > 65):
                fivehundred += 1
            elif(radius <= 65 and radius >55):
                hundred += 1
            elif(radius <= 55 and radius>50):
                fifty += 1
            else:
                ten +=1
            src = cv.circle(src, center, radius, (255, 0, 255), 3)

    cv.imshow("Blurred", dst_thr)
    cv.imshow("Original", src)
    print(" 500 won = ",fivehundred,"\r\n",
          "100 won = ",hundred,"\r\n",
          "50 won = ",fifty,"\r\n",
          "10 won = ",ten)
    cv.waitKey(0)

if __name__ =='__main__':
    print(__doc__)
    main()