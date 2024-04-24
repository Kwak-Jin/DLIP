SET(SOURCE DLIP_21900031.cpp tinyxml2.cpp)

SET(HEADER DLIP_21900031.hpp tinyxml2.h cameraParam.h)

add_library(Include STATIC ${SOURCE} ${HEADER})
