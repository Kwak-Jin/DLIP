SET(SOURCE DLIP.cpp tinyxml2.cpp)

SET(HEADER DLIP.hpp tinyxml2.h cameraParam.h)

add_library(Include STATIC ${SOURCE} ${HEADER})
