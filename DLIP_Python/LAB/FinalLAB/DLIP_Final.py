"""
========================================================================================================================
                                                DLIP LAB Final Lab
                                          Unmanned Ground Sensor System
                                            Handong Global University
========================================================================================================================
@Author  : Jin Kwak, Yoonseok Choi
@Created : 2024.06.04
@Modified:
@Version : 0.0.0
"""
import numpy as np
from ultralytics import YOLO
import cv2 as cv
import pygame
import time
from enum import Enum

CLASS_NORTH_KOREAN = 77
CLASS_PERSON       = 0
CLASS_BICYCLE      = 1
CLASS_CAR          = 2
CLASS_MOTORCYCLE   = 3
CLASSES =[CLASS_PERSON,CLASS_BICYCLE, CLASS_CAR, CLASS_MOTORCYCLE, CLASS_NORTH_KOREAN]

DISTANCE_THRESHOLD = 2.0

class Status(Enum):
    NOK   = 0
    YOK   = 1
    FAR   = 2
    CLOSE = 3

play_list =(("mixkit-arcade-chiptune-explosion-1691.wav"),
            ("mixkit-police-siren-1641.wav"))
def main()->None:
    pygame.init()
    model_detect = YOLO('yolov8s.pt')
    # model_depth  = YOLO('yolov8s.pt')
    pygame.init()
    explosion_sound = pygame.mixer.Sound(play_list[0])
    siren_sound = pygame.mixer.Sound(play_list[1])

    cap = cv.VideoCapture(1)
    cv.namedWindow('DLIP_parking_test_video', cv.WINDOW_AUTOSIZE)
    status_flag = Status.NOK
    if not cap.isOpened():
        print('Cannot open camera')
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        if (cv.waitKey(1) & 0xFF == (27)):
            break
        frame_detect = frame
        frame_depth  = frame.copy()

        result_detect = model_detect.predict(frame_detect,classes=CLASSES)
        # model_depth.predict(frame_depth)
        cv.imshow('DLIP_parking_test_video', result_detect[0].plot())
        class_cpu = result_detect[0].boxes.cls.detach().cpu().numpy()
        bbox_coords_cpu = result_detect[0].boxes.xyxy.to('cpu').numpy()

        for bbox_idx in range(len(bbox_coords_cpu)):
            bbox = bbox_coords_cpu[bbox_idx]
            if class_cpu[bbox_idx] == CLASS_NORTH_KOREAN:
                middle_point = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
                print(middle_point)
                status_flag = Status.FAR
                # Todo 1. Find the Depth of the point
                # Todo 2. Check the threshold and give either Status.FAR or Status.CLOSE
                break

        if status_flag == Status.NOK:
            continue
        elif status_flag == Status.FAR:
            print(play_list[0])
            siren_sound.play()
            time.sleep(3)
            pygame.mixer.music.set_volume(1.0)
            # while pygame.mixer.music.get_busy():
            #     pygame.time.Clock().tick(10)
        elif status_flag == Status.CLOSE:
            explosion_sound.play()
            time.sleep(1)
            # while pygame.mixer.music.get_busy():
            #     pygame.time.Clock().tick(10)
        status_flag = Status.NOK

    pygame.quit()
    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    print(__doc__)
    main()