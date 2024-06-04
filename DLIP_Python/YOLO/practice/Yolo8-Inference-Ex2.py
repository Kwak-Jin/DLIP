#########################################################
# YOLO v8  Tutorial : Prediction  Ex2
#
# Load Pretrained Model and Display the Annoted Results (multiple images)
#
#########################################################


from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Inference Source - multiple images
# Perform object detection on images using the model
results = model(['bus.jpg', 'bus2.jpg'])  # return a list of Results objects

# View results
for i, r in enumerate(results):
    # Plot results image
    dst = r.plot()  # return BGR-order numpy array
    cv2.imshow("r.plot", dst)

    # Save results to disk
    r.save(filename=f'results{i}.jpg')

cv2.waitKey(0)

