/**
* @about LAB2 Dimension measurement with 2D Camera
* @author Jin Kwak / 21900031, Mantec Ignacio / 22320052
* @created 2024.04.09
* @modified 2024.
*/
#include <iostream>
#include <opencv.hpp>
#include <string>
#include "cameraParam.h"

enum PARAM{
	LENGTH  = 1,
	HEIGHT  = 2,
	WIDTH   = 3,
	N_PARAM = 3,
};

enum CAMERA_FRAME{
	X       = 0,
	Y       = 1,
	Z       = 2,
	N_CAM   = 3,
};

enum PIXEL_FRAME{
	U       = 0,
	V       = 1,
	N_PIX   = 2
};

inline double GetVolume(double* X){
	double Area = 1;
	for(int idx = 0; idx<N_PARAM; idx++) Area*= X[idx];
	return Area;
}

double BigRectSize[N_PARAM]   = {0,};
double SmallRectSize[N_PARAM] = {0,};

int main(){
	// Camera Calibration Data
	cameraParam param("iPhone13.xml");
	cv::Mat src           ;
	cv::Mat undistortedSrc;
	undistortedSrc = param.undistort(src);
	return 0;
}