"""
========================================================================================================================
                                        DLIP Edge, Circle Detection using openCV
                                               Handong Global University
========================================================================================================================

@author: Jin Kwak 21900031
@Created:   2024.04.19
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def main():
    src = cv.imread('coins.png')
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    for i in range(5):
        gray = cv.medianBlur(gray, 3)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=1, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # Draw circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # Draw circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 0), 3)
    # Plot images
    titles = ['Original with Circle Detected']
    srcPlt=cv.cvtColor(src,cv.COLOR_BGR2RGB)
    plt.imshow(srcPlt)
    plt.title(titles)
    plt.xticks([]),plt.yticks([])
    plt.show()

if __name__ == '__main__':
    print(__doc__)
    main()