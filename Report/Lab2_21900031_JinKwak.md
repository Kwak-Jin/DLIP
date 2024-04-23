# LAB 2: Dimension Measurement with 2D camera

**Date:**  2024/04/15

**Author:**  Jin Kwak 21900031

**Github:** repository link  (if available)

**Demo Video:** Youtube link (if available)

---

# Introduction
## 1. Objective
**Goal**: Measuring the whole dimension (length, width and height) of a rectangular boxes with an iPhone and write an image processing algorithm for an accurate volume measurement of the small object.

####  Problem Conditions

- Measure the 3 dimensions (L, W, H) of a small rectangular object.
- Assume the exact width of the target object is known.
- The accuracy of the object should be within 3mm.
- 2D camera sensor only.





## 2. Preparation

Write a list of HW/SW  configuration, installation, dataset download

### Software Installation

- OpenCV 4.80,  CLion

### Dataset




# Algorithm

## 1. Overview

This is where your *concise* flow chart goes (if necessary). 

Also, another diagram (block diagram, dataflow diagram etc) can be used if it can explain the overview of the algorithm.



## 2. Procedure

### Histogram Analysis

The input image is analyzed with a histogram to understand the distribution of intensity values. As seen in the histogram in Figure 1(b), the bright component of objects can be segmented from mostly dark backgrounds. 

1. Gray-scale conversion
2. Warp perspective to make the projection appropriate With the reference object (Take the photo of the object and the reference on the same height)
3. Detecting edges or corners of both reference(credit card) and our object
4. Pixel to mm conversion of the lines and Calculate the area
5. Get another Width and Height to get the volume



####  Discussion (To think about) & Assumptions & Constraints

1.  Parallel Edges should be same in real life, but in pixels they may be different! 
2.  Credit card is not at the same height so that we cannot adjust the scale(pixel to mm) of credit card to our objects so we chose to place the credit card at the Same height.
3. 







### Filtering

Since there are visible salt noises on the input image, a median filter is applied. 

Explain what you did and why you did it. Also, explain with output images or values.



### Thresholding and Morphology

Explain what you did and why you did it. Also, explain with output images or values.





# Result and Discussion

## 1. Final Result

The result of automatic part segmentation is shown with contour boxes in Figure 00. Also, the counting output of each nut and bolt are shown in Figure 00.







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



![image-20240421015418848](C:\Users\jinkwak\AppData\Roaming\Typora\typora-user-images\image-20240421015418848.png)





---

# Appendix

Your codes go here.



**Please make the main() function as concise with high readability.**

-   It's not a good idea to write all of your algorithms within the main() function.

-   Modulize your algorithms as functions.

-   You can define your functions within your library/header 

**Write comments to  briefly describe what each function/line does**

-   It is a good practice to describe the code  with comments.
