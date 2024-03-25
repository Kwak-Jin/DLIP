# LAB Report Template

This is an example report template to help students write a concise and organized report. But you do not need to follow the exact format of this template, as long as you write a good quality report. 





# LAB: Lab Title Goes Here



**Date:**  2023-Aug-21

**Author:**  Handong Kim  20220000

**Github:** repository link  (if available)

**Demo Video:** Youtube link (if available)

---



# Introduction
## 1. Objective
Briefly explain the purpose of this lab.

**Goal**: Count the number of nuts & bolts of each size for a smart factory automation

There are 2 different size bolts and 3 different types of nuts. You are required to segment the object and count each part of 

* Bolt M5
* Bolt M6
* Square Nut M5 etc..





## 2. Preparation

Write a list of HW/SW  configuration, installation, dataset download

### Software Installation

- OpenCV 3.83,  Visual Studio 2021

- CUDA 10.1, cudatoolkit 10.1, Python 3.8.5, Pytorch 1.6.0, Torchvision 0.7.0

  

### Dataset

Description of datasets  goes here

**Dataset link:** [Download the test image](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB_grayscale/Lab_GrayScale_TestImage.jpg)






# Algorithm

## 1. Overview

This is where your *concise* flow chart goes (if necessary). 

Also, another diagram (block diagram, dataflow diagram etc) can be used if it can explain the overview of the algorithm.

<img src="https://user-images.githubusercontent.com/38373000/229727508-7d451c33-35c5-4cee-9f1e-10a4d10c21a7.png" alt="image" style="zoom:80%;" />



## 2. Procedure

### Histogram Analysis

The input image is analyzed with a histogram to understand the distribution of intensity values. As seen in the histogram in Figure 1(b), the bright component of objects can be segmented from mostly dark backgrounds. 

Explain what you did and why you did it. Also, explain with output images or values.

<img src="https://user-images.githubusercontent.com/38373000/229730944-d29b2e9f-f704-42e1-a410-6b9bda78e5fe.png" alt="image" style="zoom:30%;" />







### Filtering

Since there are visible salt noises on the input image, a median filter is applied. 

Explain what you did and why you did it. Also, explain with output images or values.



### Thresholding and Morphology

Explain what you did and why you did it. Also, explain with output images or values.





# Result and Discussion

## 1. Final Result

The result of automatic part segmentation is shown with contour boxes in Figure 00. Also, the counting output of each nut and bolt are shown in Figure 00.



<img src="https://user-images.githubusercontent.com/38373000/226501321-dcb79a67-fffc-4e8d-94f5-3b12e9868f07.png" alt="img" style="zoom:50%;" />



**Demo Video Embedded:** Youtube link (if available)



## 2. Discussion

Explain your results with descriptions and with numbers.

|   Items    | True | Estimated | Accuracy |
| :--------: | :--: | :-------: | :------: |
|  M5 Bolt   |  5   |     5     |   100%   |
|  M6 Bolt   |  10  |     9     |   90%    |
| M6 Hex Nut |  10  |     9     |   90%    |



Since this project objective is to obtain a detection accuracy of 80% for each item, the proposed algorithm has achieved the project goal successfully.





# Conclusion

Summarize the project goal and results.

Also, it suggests ways to improve the outcome.









---

# Appendix

Your codes go here.



**Please make the main() function as concise with high readability.**

-   It's not a good idea to write all of your algorithms within the main() function.

-   Modulize your algorithms as functions.

-   You can define your functions within your library/header 

**Write comments to  briefly describe what each function/line does**

-   It is a good practice to describe the code  with comments.
