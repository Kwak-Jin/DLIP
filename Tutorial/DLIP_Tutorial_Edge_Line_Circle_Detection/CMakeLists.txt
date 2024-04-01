set(Canny DLIP_Tutorial_Canny.cpp)
set(Hough DLIP_Tutorial_Hough.cpp)
set(Edge DLIP_Tutorial_Corner_Detection.cpp)
#HoughCircle

set(Optical DLIP_Tutorial_OpticalFlow.cpp)

add_executable(Canny_Detect ${Canny})
add_executable(Hough_Detect ${Hough})
add_executable(EdgeDetection ${Edge})
add_executable(OpticalFlow ${Optical})