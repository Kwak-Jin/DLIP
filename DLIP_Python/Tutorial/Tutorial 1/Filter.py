"""
========================================================================================================================
                                               DLIP Filter using OpenCV
                                               Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.19
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv.imread('HGU_logo.jpg')
    dstB = cv.blur(img, (5,5))
    dstG = cv.GaussianBlur(img, (5,5), 0)
    dstM = cv.medianBlur(img, 5)
    cv.imshow('img', img)
    cv.imshow('dstB', dstB)
    cv.imshow('dstG', dstG)
    cv.imshow('dstM', dstM)
    cv.waitKey(0)

if __name__ == '__main__':
    print(__doc__)
    main()