import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def main():
    # a simple numpy test
    a = np.array([1,2,3])
    print(a*a)

    # Load image
    img = cv.imread('HGU_logo.jpg')

    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()