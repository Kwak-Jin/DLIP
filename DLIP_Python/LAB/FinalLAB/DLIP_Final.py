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

ENUM_DETECT_NOK = 0
ENUM_DETECT_FAR = 1
ENUM_DETECT_CLO = 2
def main()->None:
    pygame.init()
    pygame.mixer.music.load("D:/DLIP/DLIP_Python/LAB/FinalLAB/mixkit-arcade-chiptune-explosion-1691.wav",
                            "")

    model = YOLO()
    isStop =False
    cap = cv.VideoCapture(0)
    cv.namedWindow('DLIP_parking_test_video', cv.WINDOW_AUTOSIZE)
    status_flag = ENUM_DETECT_NOK

    while isStop == False:
        ret, frame = cap.read()
        if not ret or (cv.waitKey(1) & 0xFF == (' ')):
            isStop = True
        if status_flag != ENUM_DETECT_NOK:
            pygame.mixer.music.play()
            pygame.mixer.music.set_volume(1.0)
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

    pygame.quit()


if __name__ == '__main__':
    print(__doc__)
    main()