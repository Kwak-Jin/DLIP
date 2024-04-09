#include <iostream>
#include <opencv.hpp>
#include "cameraParam.h"

using namespace cv;

int main(){
	cameraParam param("calibTest.xml");
	Mat src = imread("D:\\DLIP\\Tutorial\\DLIP_Tutorial_Cam_calibration\\calibTest.jpg");
	Mat dst = param.undistort(src);
	
	imshow("src", src);
	imshow("dst", dst);
	 
	waitKey(0);
	return 0;
}
