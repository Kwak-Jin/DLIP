cmake_minimum_required(VERSION 3.25)
project(DLIP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 사전에 OpenCV_DIR라는 이름으로 OpenCV 라이브러리 위치 지정 필요 (환경 변수)
find_package(OpenCV REQUIRED)
set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}/opencv2)

include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

file(COPY Image DESTINATION ${CMAKE_BINARY_DIR})

#Always make subdirectory for the associated folder e.g. assignments/LAB/projects
#add_subdirectory(Assignment)
add_subdirectory(Include)
link_directories(Include)
include_directories(Include)
link_libraries(Include)
add_subdirectory(Tutorial)
add_subdirectory(LAB)
add_subdirectory(Test)
add_subdirectory(ForFun)
