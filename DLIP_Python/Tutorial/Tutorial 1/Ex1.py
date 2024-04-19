"""
========================================================================================================================
                                        DLIP Exercise 1 (Tutorial 1) Python OpenCV
                                               Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.19
"""
import cv2 as cv
import numpy as np
thresh = 120
def main():
    src =cv.imread("rice.png")
    bsrc = cv.blur(src,(3,3))
    ret, dst = cv.threshold(bsrc,thresh,255,cv.THRESH_BINARY)
    cv.imshow("Output",dst)
    cv.waitKey(0)

if __name__ == '__main__':
    print(__doc__)
    main()