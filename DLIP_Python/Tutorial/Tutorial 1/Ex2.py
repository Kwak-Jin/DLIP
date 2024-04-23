"""
========================================================================================================================
                                        DLIP Exercise 2 (Tutorial 1) Python OpenCV
                                               Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.19
"""
import cv2 as cv
import numpy as np

def main():
    src = cv.imread('color_ball.jpg')
    dst = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    dst = cv.inRange(dst, (200, 90, 120), (215, 95, 140))
    
    cv.imshow('dst', dst)
    cv.waitKey(0)

if __name__ == '__main__':
    print(__doc__)
    main()