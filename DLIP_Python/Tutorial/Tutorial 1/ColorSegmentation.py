"""
========================================================================================================================
                                                DLIP Color Segmentation
                                               Handong Global University
========================================================================================================================
@Author: Jin Kwak
@Date: 2024.04.19
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def main():
    # Open Image in RGB
    img = cv.imread('TrafficSign1.png')

    # Convert BRG to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # matplotlib: cvt color for display
    imgPlt = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Color InRange()
    lower_range = np.array([90,128,0])
    upper_range = np.array([215,255,255])
    dst_inrange = cv.inRange(hsv, lower_range, upper_range)

    # Mask selected range
    mask = cv.inRange(hsv, lower_range, upper_range)
    dst = cv.bitwise_and(hsv,hsv, mask= mask)

    # Plot Results
    titles = ['Original ', 'Mask','Inrange']
    images = [imgPlt, mask, dst]

    for i in range(3):
        plt.subplot(1,3,i+1),plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

if __name__ == '__main__':
    print(__doc__)
    main()