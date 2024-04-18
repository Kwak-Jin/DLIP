import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# a simple numpy test
a = np.array([1,2,3])
print(a*a)

# Load image
img = cv.imread('HGU_logo.jpg')

# Display Image
# cv.namedWindow('source', cv.WINDOW_AUTOSIZE)
cv.imshow('source',img)
cv.waitKey(0)