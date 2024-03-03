cmake_minimum_required(VERSION 3.25)
project(PRACTICE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 사전에 OpenCV_DIR라는 이름으로 OpenCV 라이브러리 위치 지정 필요
find_package(OpenCV REQUIRED PATHS $ENV{OpenCV_DIR})

include_directories(${OpenCV_INCLUDE_DIR})
link_libraries(${OpenCV_LIBS})

set(SOURCE1 practice1.cpp)
set(SOURCE2 practice2.cpp)

add_executable(practice1 ${SOURCE1})
add_executable(practice2 ${SOURCE2})

#Always make subdirectory for the associated folder e.g. assignments/LAB/projects
add_subdirectory(1_21900031_곽진)