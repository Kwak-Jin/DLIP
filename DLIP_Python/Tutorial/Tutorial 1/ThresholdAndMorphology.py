"""
DLIP Python
Author: Jin Kwak 21900031
Created: 2024.04.19
Brief Description: Thresholds and Morphology in Python
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def main():
    img = cv.imread("HGU_logo.jpg")
    cv.imshow("Original", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thVal = 127
    # Apply Thresholding
    ret, thresh1 = cv.threshold(img, thVal, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, thVal, 255, cv.THRESH_BINARY_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV']

    cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel = np.ones((5, 5), np.uint8)

    # Morphology
    erosion  = cv.erode (thresh1, kernel, iterations=1)
    dilation = cv.dilate(thresh1, kernel, iterations=1)
    opening  = cv.morphologyEx(thresh2, cv.MORPH_OPEN, kernel)
    closing  = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, kernel)

    # Plot results
    titles = ['Original ', 'Opening', 'Closing','Erosion', 'dilation']
    images = [img, opening, closing, erosion,dilation]

    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    plt.show()

if __name__ == '__main__':
    print(__doc__)
    main()