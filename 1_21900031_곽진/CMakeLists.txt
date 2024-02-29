cmake_minimum_required(VERSION 3.25)
project(YOLOv8 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)


# 사전에 OpenCV_DIR라는 이름으로 OpenCV 라이브러리 위치 지정 필요
find_package(OpenCV REQUIRED PATHS $ENV{OpenCV_DIR})

include_directories( ${OpenCV_INCLUDE_DIR})
link_libraries(${OpenCV_LIBS})

