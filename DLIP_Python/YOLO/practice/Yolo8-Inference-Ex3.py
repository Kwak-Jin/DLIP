#########################################################
# YOLO v8  Tutorial : Prediction  Ex3
#
# Stream Video Prediction
#
# This script will run predictions on each frame of the video
# visualize the results, and display them in a window.
# The loop can be exited by pressing 'q'.
#########################################################


import cv2 as cv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('BEST.pt')

# Open the video camera no.0
cap = cv.VideoCapture(0)

# If not success, exit the program
if not cap.isOpened():
    print('Cannot open camera')


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()

