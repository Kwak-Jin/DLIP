"""
========================================================================================================================
                                        DLIP LAB 4 Python CNN Object Detection
                                            Parking Management System
                                            Handong Global University
========================================================================================================================
@Author  : Jin Kwak
@Created : 2024.05.24
@Modified: 2024.06.04
@Version : Final
"""
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from shapely.geometry import Polygon

x_ROI   = 0
w_ROI   = 1280              # Video Width
y_ROI   = int(720/2) - 80  # Video Height Boundary
h_ROI   = int(720/2) + 90  # Video Height Boundary

ENUM_CAR  = 2 # Car
ENUM_MOT  = 3 # MotorCycle
ENUM_BUS  = 5 # Bus
ENUM_TRU  = 7 # Truck

lower_y = 432
upper_y = 323
green   = (0  , 255, 0  )
pink    = (255, 0  , 255)
parking_spaces = (((56  , upper_y), (140 , upper_y), (85  , lower_y), (0   , lower_y)),  # Parking #1
                  ((140 , upper_y), (255 , upper_y), (196 , lower_y), (85  , lower_y)),  # Parking #2
                  ((255 , upper_y), (350 , upper_y), (308 , lower_y), (196 , lower_y)),  # Parking #3
                  ((350 , upper_y), (452 , upper_y), (422 , lower_y), (308 , lower_y)),  # Parking #4
                  ((452 , upper_y), (540 , upper_y), (530 , lower_y), (422 , lower_y)),  # Parking #5
                  ((540 , upper_y), (634 , upper_y), (635 , lower_y), (530 , lower_y)),  # Parking #6
                  ((634 , upper_y), (727 , upper_y), (742 , lower_y), (635 , lower_y)),  # Parking #7
                  ((727 , upper_y), (818 , upper_y), (848 , lower_y), (742 , lower_y)),  # Parking #8
                  ((818 , upper_y), (912 , upper_y), (954 , lower_y), (848 , lower_y)),  # Parking #9
                  ((912 , upper_y), (1000, upper_y), (1050, lower_y), (954 , lower_y)),  # Parking #10
                  ((1000, upper_y), (1090, upper_y), (1156, lower_y), (1050, lower_y)),  # Parking #11
                  ((1090, upper_y), (1200, upper_y), (1264, lower_y), (1156, lower_y)),  # Parking #12
                  ((1200, upper_y), (1280, upper_y), (1280, lower_y), (1264, lower_y)))  # Parking #13

def main()-> None:
    cap = cv.VideoCapture('DLIP_parking_test_video.avi')
    cv.namedWindow('DLIP_parking_test_video', cv.WINDOW_AUTOSIZE)
    model = YOLO('yolov8s.pt')
    isStop =False
    frame_number = 0
    while isStop == False:
        ret, frame = cap.read()
        if not ret or (cv.waitKey(1) & 0xFF == (' ')):
            isStop = True

        result = model.predict(source=frame, save=False, save_txt=False)
        bbox_coords_cpu = result[0].boxes.xyxy.to('cpu').numpy()
        class_cpu       = result[0].boxes.cls.detach().cpu().numpy()

        occupied_space = 0
        parking_idx = []
        for bbox_idx in range(len(bbox_coords_cpu)):
            if (((bbox_coords_cpu[bbox_idx,3]-bbox_coords_cpu[bbox_idx,1])/2)< lower_y):
                if class_cpu[bbox_idx] == ENUM_CAR or class_cpu[bbox_idx] == ENUM_BUS or class_cpu[bbox_idx] == ENUM_TRU:
                    bounding_box_coordinates = ((bbox_coords_cpu[bbox_idx,0], bbox_coords_cpu[bbox_idx,1]),
                                                (bbox_coords_cpu[bbox_idx,0], bbox_coords_cpu[bbox_idx,3]),
                                                (bbox_coords_cpu[bbox_idx,2], bbox_coords_cpu[bbox_idx,3]),
                                                (bbox_coords_cpu[bbox_idx,2], bbox_coords_cpu[bbox_idx,1]),
                                                (bbox_coords_cpu[bbox_idx,0], bbox_coords_cpu[bbox_idx,1]))
                    parking_location = calculate_intersection_area(parking_spaces, bounding_box_coordinates)
                    if parking_location in parking_idx or parking_location == 0:
                        continue
                    parking_idx.append(parking_location)
                    occupied_space += 1
        parking_idx.sort()
        for parking in range(1,14):
            if parking in parking_idx:
                continue
            cv.fillConvexPoly(frame,np.array(list(parking_spaces[parking-1])),green)
        cv.putText(frame, "Parked Location "+str(parking_idx)            ,(0, 50),cv.FONT_HERSHEY_PLAIN,2,pink,2)
        cv.putText(frame, 'Free Parking Spaces '+str(13 - occupied_space),(0,150),cv.FONT_HERSHEY_PLAIN,2,pink,2)
        cv.imshow('DLIP_parking_test_video', result[0].plot())
        if (frame_number)== 2500 or frame_number == 1500:
            cv.imwrite("result"+str(frame_number)+".jpg", frame)
            if frame_number >5000:
                break
        frame_number += 1
    cv.destroyAllWindows()
    cap.release()

def calculate_intersection_area(parking_coordinates:tuple, bbox_coordinates:tuple) -> int:
    rectangle = Polygon(bbox_coordinates)
    idx = 1
    num = 0
    for coordinates in parking_coordinates:
        trapezoid = Polygon(coordinates)
        intersection = trapezoid.intersection(rectangle)
        # print("Area")
        # print(intersection.area)
        if intersection.area > 4000:
            num = idx
        idx += 1
    return num

if __name__ == '__main__':
    print(__doc__)
    main()
