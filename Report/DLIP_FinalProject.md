# LAB: Unmanned Ground Sensor System

**Date:**  2024.06.11

**Author:**  Jin Kwak/ 21900031, Yoonseok Choi/22100747

**Course:** Image Processing with Deep Learning

**[Github](https://github.com/Kwak-Jin/DLIP)**

**[Demo Video](https://youtu.be/1wIuSlom4ok)**

---

# Introduction
## 1. Objective
This lab is to replace manpower for alert duty in military using mono-camera.

For human&camera replacement, the system should be able to detect the depth and class of an object(North Korean) using a mono camera.

**Goal**: Depth/Object detection in the military region 

**Reason For the Topic**

- Due to low birth rate, reduction in military power.
- Guard Duty Burden
- Reduced Combat power due to non-combat(Alert Duty) missions(Severe Weather)

### Problem Conditions

- Use Python OpenCV (*.py)
- Deep Learning Model
  - Mono-Depth Estimation: []()
  - Object Detection: [YOLO V8](https://www.ultralytics.com/yolo)
- image
  - Curvature edge

## 2. Preparation

### Software Installation

```shell
```



### Dataset

Download the test images of

- 


# Algorithm

## 1. Overview

The flow of the program is described in figure 1. Image preprocessing parts are selected in a rectangle box.

<p align='center'><img src="..\Report_image\LAB3\FlowChart.png" alt="FlowChart" style="zoom:120%;" /> Figure 1. Flowchart of the program </p>

1. Select ROI: As metal plate is always located at the left side of the image, ROI is applied.
2. Convert Color: For most of the algorithms, the rgb image is converted to gray-scale
3. Filter: Applied for noise reduction. Both Median and Gaussian Blur.
4. Image Enhancing: The image has low intensity in gray-scale therefore, the equalization of histogram technique is used.
5. Edge Detection: To detect the curvature of the metal plate edge, Canny Edge detection is used.
6. **Edge Coordinates Detection**: For Quadratic Regression(In the perspective of Least Square Method), only the metal plate edge coordinates should be collected therefore, this part is the most significant process in the lab. For cross checking, the edge coordinates are collected from the left side and from the right side each.
7. Quadratic regression: This is to mathematically approximate the curvature. This is done by `np.polyfit`. As the curvature is parabolic, the degree of the regression is 2. After proper regression, the maximum point of the curvature is obtained and this can be applied to the original image. Then calculation of the level is applied.

## 2. Procedure

### Region of Interest(ROI)

Since the metal sheet's position is fixed (left side of the image as below), I selected  ROI as Table 1.

| **Position** | **Coordinate** |
| :----------: | -------------- |
|   Top-Left   | (125, 200)     |
|  Top-Right   | (700, 200)     |
| Bottom-Left  | (125,1080)     |
| Bottom-Right | (700,1080)     |

<p align = 'center'><img src="..\Report_image\LAB3\LV1.png" alt="LV1" style="zoom:60%;" /> Figure 2. Example Image of the rolling metal sheet </p>

<p align= 'center'><img src="..\Report_image\LAB3\Report_LV2.png" alt="Report_LV2" style="zoom:70%;" /> Figure 3. Example Image with ROI applied </p>

### Histogram Analysis (Image Enhancing using histogram equalization)

The original image's histogram looks like Figure 4.

<p align='center'><img src="..\Report_image\LAB3\og_histogram.png" alt="og_histogram" style="zoom:120%;" /> Figure 4. Origianl Image Histogram  </p>

After preprocessing, the histogram will look like Figure 5. Due to iteration of filters(both linear and non-linear), the shape of the graph changed as below.

<p align='center'><img src="..\Report_image\LAB3\AfterPreprocess_histogram.png" alt="AfterPreprocess_histogram" style="zoom:120%;" /> Figure 5. Preprocessed Image Histogram </p>

Finally, after equalization of histogram, the final histogram is made in Figure 6.

<p align='center'><img src="..\Report_image\LAB3\AfterEqualization_histogram.png" alt="AfterPreprocess_histogram" style="zoom:120%;" /> Figure 6. Equalized Image Histogram </p>

### Filtering

There are 2 filters used in the program.

- Median Filter
  - Kernel size 5
  - Iteration 20
  - Reason: To remove salt and pepper noise and high intensity point
- Gaussian Noise
  - Kernel size 5
  - Iteration 2
  - Reason: To reduce high light intensity point to average

### Thresholding and Morphology

None of thresholding and morphology technique is used in the program

### Edge Detection

- Canny Edge Detection is used in the program to detect the edges of the metal plate.
- Thresholding for the Canny Edge is adjusted for accuracy

###  Quadratic Regression

- For the quadratic regression, each coordinate of detected edge is selected.
- For the accurate selection of the rolling metal plate edge, discontinuous points are rejected.
- From left side, coordinates are collected from the bottom
- From right side, coordinates are collected from the top
- The selected coordinates are regressed using `np.polyfit()` with the degree of 2.
- Maximum point of the coordinate is selected and used for the tension level.

# Result and Discussion

## 1. Final Result

|             | Level 1                                                      | Level 2                                                      | Level 3                                                      |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Challenging | <img src="..\Report_image\LAB3\Report_Challenging1.png" alt="Report_Challenging1" style="zoom:80%;" /> | <img src="..\Report_image\LAB3\Report_Challenging2.png" alt="Report_Challenging2" style="zoom:80%;" /> | <img src="..\Report_image\LAB3\Report_Challenging3.png" alt="Report_Challenging3" style="zoom:80%;" /> |
| Video       | <img src="..\Report_image\LAB3\Report_Video1.png" alt="Report_Video1" style="zoom:80%;" /> | <img src="..\Report_image\LAB3\Report_Video2.png" alt="Report_Video2" style="zoom:80%;" /> | <img src="..\Report_image\LAB3\Report_Video3.png" alt="Report_Video3" style="zoom:80%;" /> |

**[Demo Video Embedded](https://youtu.be/1wIuSlom4ok)**

## 2. Discussion

Images of different levels are tested. In some frames of video, there was false detection of level but in most of the frames, the level detection worked well.

|  List   | Accuracy [%] |
| :-----: | :----------: |
| Level 1 |    100.0     |
| Level 2 |    100.0     |
| Level 3 |    100.0     |
|  Video  |    99.482    |

Therefore, the algorithm satisfies the objectives of the lab.

# Conclusion

## Lab Purpose Achievement

The algorithm is performed with the specified objectives and verified by the videos. The most difficult part of the algorithm is Detecting the coordinates of the edge. After appropriate preprocessing, the first captured point is the part of curvature and appropriate calculation could be done.

As the industrial problem deals with the reflection of copper plate, and dark, reddish image, histogram equalization, filters, selection of region of interest techniques are used and finally successful program can be achieved.

## Improvement

1. Since the metal plate's edge can be detected with the curvature, the line can also be detected and this line will meet the maximum point(vertex of the curvature) as in Figure 7. If this can be done using `Houghlines()`, an accurate vertex can be obtained.

   <p align='center'><img src="..\Report_image\LAB3\Curvature and Line.png" alt="Curvature and Line" style="zoom:75%;" /> Figure 7. Curvature and Line </p>

2.  Another improvement can be made with memory. The FPS(Frame per second) is 30. Since the metal plate does not change its level quickly, the maximum points of the previous frames can be saved and used as filter buffers. In this case, the exponential moving average filter may be used

---

# Appendix

`Video`

```python
"""
========================================================================================================================
                            DLIP LAB 3 Python Tension Detection of Rolling Metal Sheet
                                Challenging Video(Save Video) Using append()
                                          Handong Global University
========================================================================================================================
@Author  : Jin Kwak
@Created : 2024.04.27
@Modified: 2024.05.07
@Version : 1.2
"""
import cv2 as cv
import numpy as np

N_MedianIter = 25
kernel = 5
YOK = 1
NOK = 0
level2 = 250
level3 = 120
""" ROI length and width"""
ROI_y = 200
ROI_h = 1080 - ROI_y
ROI_x = 125
ROI_w = 575


def main():
    cap = cv.VideoCapture('../../Image/LAB3_Video.mp4')
    if not cap.isOpened():
        print('Cannot open Video Capture')

    discontinuity = 50
    cv.namedWindow('Src', cv.WINDOW_AUTOSIZE)

    while True:
        # Read a new frame from video
        ret, frame = cap.read()
        if checkEnd(ret):
            break

        roi = frame[ROI_y:ROI_y + ROI_h, ROI_x:ROI_x + ROI_w]
        psrc = roi.copy()
        pre_src = preprocess(roi)

        fx1 = []
        fx = []
        xx1 = []
        xx = []
        totalX = np.zeros((ROI_w - 1), np.int16)
        for idx in range(0, ROI_w):              # col
            for tdx in range(ROI_h - 1, 0, -1):  # row
                if pre_src[tdx, idx] == 255:
                    fx.append(tdx)  # level
                    xx.append(idx)
                    if len(xx) > 1:
                        psrc = cv.circle(psrc, (xx[-1], fx[-1]), 1, (255, 0, 255), -1)
                    break
            if (len(fx) > 2) and ((np.abs(fx[-1] - fx[-2]) >= discontinuity) or (np.abs(xx[-1] - xx[-2]) >= 20)) :
                break
        for idx in range(ROI_w - 1, 0, -1):   # col
            for tdx in range(0, ROI_h - 1):   # row
                if pre_src[tdx, idx] == 255:
                    fx1.append(tdx)  # row
                    xx1.append(idx)
                    if len(xx1) > 1:
                        psrc = cv.circle(psrc, (xx1[-1], fx1[-1]), 1, (255, 0, 255), -1)
                    break
            if (idx < ROI_w - 2) and (np.abs(fx1[-1] - fx1[-2]) >= discontinuity):
                break
        xx = np.hstack([xx, xx1])
        fx = np.hstack([fx, fx1])
        for idx in range(0, ROI_w - 1):
            totalX[idx] = idx
        coeff = np.polyfit(xx, fx, 2)
        p = np.poly1d(coeff)
        # Predicted values
        y_pred = p(xx)

        rmse = np.sqrt(np.mean((fx - y_pred) ** 2))

        maxY = 0
        for x in range(0, frame.shape[1] - 1):
            newY = coeff[0] * (x - ROI_x) ** 2 + coeff[1] * (x - ROI_x) + coeff[2] + ROI_y
            if newY > maxY:
                maxY = newY
            newY = int(newY)
            frame = cv.circle(frame, (x, newY), 1, (255, 255, 0), -1)
        """ After Polyfit  """
        if maxY < frame.shape[0] - level3 and maxY >= frame.shape[0] - level2:
            _str = "Level: 2"
        elif maxY > frame.shape[0] - level3:
            _str = "Level: 3"
        else:
            _str = "Level: 1"

        frame = drawLevelBound(frame)
        bend  = 'Score:' + str(frame.shape[0]-maxY)
        __str = "RMSE :" + str(rmse)
        cv.putText(frame, _str, ((int)(ROI_w)+400, (int)(ROI_h / 2) - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(frame, bend, ((int)(ROI_w)+400, (int)(ROI_h / 2)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv.putText(frame, __str, ((int)(ROI_w)+400, (int)(ROI_h / 2)+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Src', frame)
        cv.imshow('Canny', pre_src)
        cv.imshow("CurveFit", psrc)

    cv.destroyAllWindows()
    cap.release()

def preprocess(img: np.ndarray) -> np.ndarray:
    _dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(0, N_MedianIter):
        _dst = cv.medianBlur(_dst, kernel)
    for i in range(0, 2):
        _dst = cv.GaussianBlur(_dst, (kernel, kernel), 0)

    cv.equalizeHist(_dst, _dst)
    _dst = cv.Canny(_dst, 60, 255)
    return _dst


def checkEnd(ret: bool) -> int:
    if cv.waitKey(5) & 0xFF == 27 or not ret:
        print('Video End!')
        flag = YOK
    else:
        flag = NOK
    return flag


def drawLevelBound(img: np.ndarray) -> np.ndarray:
    cv.line(img, (0, img.shape[0] - level3), (img.shape[1], img.shape[0] - level3), (255, 0, 255), 2,
            cv.LINE_AA)  # Level 3
    cv.line(img, (0, img.shape[0] - level2), (img.shape[1], img.shape[0] - level2), (255, 255, 0), 2,
            cv.LINE_AA)  # Level 2
    return img


if __name__ == '__main__':
    print(__doc__)
    main()

```

`Image`

```python
"""
========================================================================================================================
                            DLIP LAB 3 Python Tension Detection of Rolling Metal Sheet
                                            Challenging Dataset
                                          Handong Global University
========================================================================================================================
@Author  : Jin Kwak
@Created : 2024.04.23
@Modified: 2024.05.07
@Version : 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

N_MedianIter = 25
kernel = 5
YOK = 1
NOK = 0
level2 = 250
level3 = 120
""" ROI length and width"""
ROI_y = 200
ROI_h = 1080 - ROI_y
ROI_x = 125
ROI_w = 575
def main():
    src = cv.imread("../../Image/Challenging_Dataset/LV3.png")
    discontinuity = 20
    roi = src[ROI_y:ROI_y + ROI_h, ROI_x:ROI_x + ROI_w]
    psrc = roi.copy()
    pre_src = Preprocess(psrc)
    cv.line(src, (0,src.shape[0]-120), (src.shape[1],src.shape[0]-120), (255,0,255), 2, cv.LINE_AA) # Level 3
    cv.line(src, (0,src.shape[0]-250), (src.shape[1],src.shape[0]-250), (255,255,0), 2 ,cv.LINE_AA) # Level 2
    fx1 = []
    fx = []
    xx1 = []
    xx = []
    totalX = np.zeros((ROI_w - 1), np.int16)
    for idx in range(0, ROI_w):  # col
        for tdx in range(ROI_h - 1, 0, -1):  # row
            if pre_src[tdx, idx] == 255:
                fx.append(tdx)  # level
                xx.append(idx)
                if len(xx) > 1:
                    psrc = cv.circle(psrc, (xx[-1], fx[-1]), 1, (255, 0, 255), -1)
                break
        if (len(fx) > 2) and ((np.abs(fx[-1] - fx[-2]) >= discontinuity) or (np.abs(xx[-1] - xx[-2]) >= 20)):
            break
    for idx in range(ROI_w - 1, 0, -1):  # col
        for tdx in range(0, ROI_h - 1):  # row
            if pre_src[tdx, idx] == 255:
                fx1.append(tdx)  # row
                xx1.append(idx)
                if len(xx1) > 1:
                    psrc = cv.circle(psrc, (xx1[-1], fx1[-1]), 1, (255, 0, 255), -1)
                break
        if (idx < ROI_w - 2) and (np.abs(fx1[-1] - fx1[-2]) >= discontinuity):
            break

    xx = np.hstack([xx, xx1])
    fx = np.hstack([fx, fx1])
    for idx in range(0, ROI_w - 1):
        totalX[idx] = idx
    coeff = np.polyfit(xx, fx, 2)
    if (coeff[0]>0):
        coeff[0] = -coeff[0]
    p = np.poly1d(coeff)
    # Predicted values
    y_pred = p(xx)

    rmse = np.sqrt(np.mean((fx - y_pred) ** 2))

    maxY = 0
    for x in range(0, src.shape[1] - 1):
        newY = coeff[0] * (x - ROI_x) ** 2 + coeff[1] * (x - ROI_x) + coeff[2] + ROI_y
        if newY > maxY:
            maxY = newY
        newY = int(newY)
        src = cv.circle(src, (x, newY), 1, (255, 255, 0), -1)
    """ After Polyfit  """
    if maxY < src.shape[0] - level3 and maxY >= src.shape[0] - level2:
        _str = "Level: 2"
    elif maxY > src.shape[0] - level3:
        _str = "Level: 3"
    else:
        _str = "Level: 1"

    # src = drawLevelBound(src.shape[0] - src)
    src = drawLevelBound(src)
    bend = 'Score:' + str(src.shape[0]-maxY)
    __str = "RMSE :" + str(rmse)
    cv.putText(src, _str, ((int)(ROI_w) + 400, (int)(ROI_h / 2) - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(src, bend, ((int)(ROI_w) + 400, (int)(ROI_h / 2)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv.putText(src, __str, ((int)(ROI_w)+400, (int)(ROI_h / 2)+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow('Src', src)
    cv.imshow('Canny', pre_src)
    cv.imshow("CurveFit", psrc)
    cv.waitKey(0)


"""
img: BGR image
return: Grayscale image after preprocessing (Filtering, Equalize Histogram ...) 
"""
def Preprocess(img: np.ndarray) -> np.ndarray:
    _dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(0,20):
        _dst = cv.medianBlur(_dst,5)
    for i in range(0,3):
        _dst = cv.GaussianBlur(_dst, (5, 5), 0)
    cv.equalizeHist(_dst, _dst)

    _dst = cv.Canny(_dst,40,255)
    return _dst

def drawLevelBound(img: np.ndarray) -> np.ndarray:
    cv.line(img, (0, img.shape[0] - level3), (img.shape[1], img.shape[0] - level3), (255, 0, 255), 2,
            cv.LINE_AA)  # Level 3
    cv.line(img, (0, img.shape[0] - level2), (img.shape[1], img.shape[0] - level2), (255, 255, 0), 2,
            cv.LINE_AA)  # Level 2
    return img

if __name__ == '__main__':
    print(__doc__)
    main()
```



