# LAB 1: Grayscale Image Segmentation



**Date:**  2024-03-24

**Author:**  Kwak Jin, 21900031

**Github:** https://github.com/Kwak-Jin/DLIP

**Demo Video:**  -

---

# Introduction
## 1. Objective
This project is done to count the number of nuts&bolts of each size for smart factory only using image processing.

**Goal**: Count the number of nuts & bolts of each size for a smart factory automation

There are 2 different size bolts and 3 different types of nuts. You are required to segment the object and count each part of 

* Bolt M5
* Bolt M6
* Square Nut M5 etc..

## 2. Preparation

Write a list of HW/SW  configuration, installation, dataset download

### Software Installation

- OpenCV 4.8.0
- Jetbrains CLion (IDE)

### Dataset

​	The data set is shown as the following:

<img src="..\cmake-build-debug\LAB\Lab_GrayScale_TestImage.jpg" alt="Lab_GrayScale_TestImage" style="zoom:50%;" />

​	As in the histogram, the intensity is very concentrated near lower half. In other words, the contrast is relatively low and is not easy to distinguish between each objects. 

<img src="..\Report_image\LAB1\Histogram.JPG" alt="Histogram_Original" style="zoom: 50%;" />

Therefore, 

**Dataset link:** [Download the test image](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB_grayscale/Lab_GrayScale_TestImage.jpg)




# Algorithm

## 1. Overview

This is where your *concise* flow chart goes (if necessary). 

Also, another diagram (block diagram, dataflow diagram etc) can be used if it can explain the overview of the algorithm.



## 2. Procedure

### Histogram Analysis

The input image is analyzed with a histogram to understand the distribution of intensity values. As seen in the histogram in Figure 1(b), the bright component of objects can be segmented from mostly dark backgrounds. 

Explain what you did and why you did it. Also, explain with output images or values.



### Filtering

Since there are visible salt noises on the input image, a median filter is applied. 

Explain what you did and why you did it. Also, explain with output images or values.



### Thresholding and Morphology

Explain what you did and why you did it. Also, explain with output images or values.





# Result and Discussion

## 1. Final Result

The result of automatic part segmentation is shown with contour boxes in Figure 00. Also, the counting output of each nut and bolt are shown in Figure 00.

<img src="https://user-images.githubusercontent.com/38373000/226501321-dcb79a67-fffc-4e8d-94f5-3b12e9868f07.png" alt="img" style="zoom:50%;" />

## 2. Discussion

Explain your results with descriptions and with numbers.

|   Items    | True | Estimated | Accuracy |
| :--------: | :--: | :-------: | :------: |
|  M5 Bolt   |  5   |     5     |   100%   |
|  M6 Bolt   |  10  |     9     |   90%    |
| M6 Hex Nut |  10  |     9     |   90%    |

Since this project objective is to obtain a detection accuracy of 80% for each item, the proposed algorithm has achieved the project goal successfully.

# Conclusion

This project is done to  figure out appropriate classic image processing techniques for correct image segmentation. Some of the followings are used to segment each bolts and nuts such as **Spatial Filtering**, **Morphology(Opening and Closing)**,**Contouring**, and **Thresholding**.  

---

# Appendix

```c++
```

**Please make the main() function as concise with high readability.**

-   It's not a good idea to write all of your algorithms within the main() function.

-   Modulize your algorithms as functions.

-   You can define your functions within your library/header 

**Write comments to  briefly describe what each function/line does**

-   It is a good practice to describe the code  with comments.
