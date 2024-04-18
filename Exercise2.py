import cv2 as cv

# Open the video camera no.0
cap = cv.VideoCapture(0)

# If not success, exit the program
if not cap.isOpened():
    print('Cannot open camera')

cv.namedWindow('MyVideo', cv.WINDOW_AUTOSIZE)

while True:
    # Read a new frame from video
    ret, frame = cap.read()

    # If not success, break loop
    if not ret:
        print('Cannot read frame')
        break

    cv.imshow('MyVideo', frame)

    if cv.waitKey(30) & 0xFF == 27:
        print('Press ESC to stop')
        break

cv.destroyAllWindows()
cap.release()
