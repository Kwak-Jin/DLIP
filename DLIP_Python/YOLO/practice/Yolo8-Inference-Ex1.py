#########################################################
# YOLO v8  Tutorial : Prediction  Ex1
#
# Load Pretrained Model and Display the Annoted Results
#
#########################################################

from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Inference Source - a single source
src = cv2.imread("bus.jpg")

# Perform object detection on an image using the model
result = model.predict(source=src, save=True, save_txt=True)  # save predictions as labels

# View result
for r in result:
    # print the Boxes object containing the detection bounding boxes
    print(r.boxes)

    # Show results to screen (not recommended)
    print("result.show()")
    r.show()

    # Plot results image
    print("result.plot()")
    dst = r.plot()  # return BGR-order numpy array
    cv2.imshow("result plot", dst)

    # Plot the original image (NParray)
    print("result.orig_img")
    cv2.imshow("result orig", r.orig_img)

# Save results to disk
r.save(filename='result.jpg')
cv2.waitKey(0)

##########################################################################################


