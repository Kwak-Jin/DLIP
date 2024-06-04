# LAB: CNN Object Detection 

# Parking Management System

**Date:**  2024.06.04

**Author:**  Jin Kwak/ 21900031

**[Github](https://github.com/Kwak-Jin/DLIP)**

**[Demo Video](https://youtu.be/1wIuSlom4ok)**

---

# Introduction
## 1. Objective
​	Vehicle counting using a CNN based object detection model.

In this lab,  the author is required to create a simple program that 

(1) counts the number of vehicles in the parking lot 

(2) display the number of available parking space.

​	For the given dataset, the maximum available parking space is 13. If the current number of vehicles is more than 13, then, the available space should display as ‘0’.

### Problem Conditions

- Use Python OpenCV (*.py)
- Use pretrained YOLO v8.

## 2. Preparation

### Software Installation

- Python 3.9
- opencv-python 4.9.0 
- PyTorch 2.2.0
- PyCharm
- Miniconda (Virtual Environment)

## 3. Procedure

- Download the test video file: [click here to download](https://drive.google.com/file/d/1d5RATQdvzRneSxvT1plXxgZI13-334Lt/view?usp=sharing)
- Need to count the number of vehicles in the parking lot for each frame
  - **DO NOT COUNT** the vehicles outside the parking spaces
  - Consider the vehicle is outside the parking area if the car's center is outside the parking space
- Make sure to not count duplicates of the same vehicle
- It should accurately display the current number of vehicle and available parking spaces
- Save the vehicle counting results in '**counting_result.txt'** file.
  - When program is executed, the 'counting_result.txt' file should be created automatically for a given input video.
  - Each line in text file('counting_result.txt') should be the pair of **frame# and number of detected car**.
  - Frame number should start from 0. ex) 0, 12 1, 12 ...
- In the report, evaluate the model performance with numbers (accuracy etc)
  - Answer File for Frame 0 to Frame 1500 are provided: [download file](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB-ParkingSpace/LAB_Parking_counting_result_answer_student_modified.txt)
- Program will be scored depending on the accuracy of the vehicle numbers
  - TA will check the Frame 0 to the last frame


# Algorithm

## 1. Overview

Flowchart of the program is described in Figure 1. The preprocessing part in the left top box is subdivided into ROI selection, dividing the parking spaces, and get Pretrained YOLO model (`yolo8s.pt`) 

<p/ align = 'center'><img src="..\Report_image\LAB4\LAB4FlowChart.png" alt="LAB4FlowChart"  /> Figure 1. Flowchart of the Program </p>

## 2. Procedure

### Preprocessing

#### Region of Interest(ROI)

​	To only count the number of vehicles in the parking lot, the ROI is selected. 

| **Position** | **Coordinate** |
| :----------: | -------------- |
|   Top-Left   |                |
|  Top-Right   |                |
| Bottom-Left  |                |
| Bottom-Right |                |

#### Divide Parking Spaces

​	The Parking Spaces are divided for the parking space occupancy check. This is defined at the start of the program with 2-Dimensional list.

```python
LowerY = 432
UpperY = 323

parking_spaces = [
    [(56, UpperY), (140, UpperY), (85, LowerY), (0, LowerY)],  # Parking #1
    [(140, UpperY), (255, UpperY), (196, LowerY), (85, LowerY)],  # Parking #2
    [(255, UpperY), (350, UpperY), (308, LowerY), (196, LowerY)],  # Parking #3
    [(350, UpperY), (452, UpperY), (422, LowerY), (308, LowerY)],  # Parking #4
    [(452, UpperY), (540, UpperY), (530, LowerY), (422, LowerY)],  # Parking #5
    [(540, UpperY), (634, UpperY), (635, LowerY), (530, LowerY)],  # Parking #6
    [(634, UpperY), (727, UpperY), (742, LowerY), (635, LowerY)],  # Parking #7
    [(727, UpperY), (818, UpperY), (848, LowerY), (742, LowerY)],  # Parking #8
    [(818, UpperY), (912, UpperY), (954, LowerY), (848, LowerY)],  # Parking #9
    [(912, UpperY), (1000, UpperY), (1050, LowerY), (954, LowerY)],  # Parking #10
    [(1000, UpperY), (1090, UpperY), (1156, LowerY), (1050, LowerY)],  # Parking #11
    [(1090, UpperY), (1200, UpperY), (1264, LowerY), (1156, LowerY)],  # Parking #12
    [(1200, UpperY), (1280, UpperY), (1280, LowerY), (1264, LowerY)]  # Parking #13
]
```

​	These coordinates represent each edge of parking spaces. As the image is taken in the view of CCTV not a bird-eye view, the parking spaces are trapezoid-shaped which is visualized in Figure 2. As there were always occupied spaces in the video, the exact parking space edges are not selected. However, the bounding box is always a rectangle without any rotation. To maximize the overlapped area, the parking spaces are carefully selected by hands to enlarge the parking space height. Furthermore, `LowerY`, `UpperY` is the average y- coordinates of the parking spaces in the y direction of pixel frame. Constant `LowerY` is later used to consider the existence of a car in the parking lot(compare with car's center)

<p align='center'><img src="D:\DLIP\Report_image\LAB4\DLIP_ParkingArea.jpg" alt="DLIP_ParkingArea" style="zoom:80%;" /> Figure 2. Divided Parking Spaces</p>



#### Python Packages

##### Ultralytics

`pip install ultralytics`

The `ultralytics` used in the program is **8.2.18**.

##### Shapely

`pip install shapely`

This package is used to calculate the overlapped area of the bounding box and pre-defined parking spaces. The maximum overlapped area is used to select which parking space the car has parked.

# Result and Discussion

## 1. Final Result

**[Demo Video Embedded](https://youtu.be/1wIuSlom4ok)**



## 2. Discussion



Therefore, the algorithm satisfies the objectives of the lab.

# Conclusion

## Lab Purpose Achievement



## Improvement



# Appendix

`Source Code`

```python
```



