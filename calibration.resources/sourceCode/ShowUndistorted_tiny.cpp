#include <iostream>
#include <opencv.hpp>
#include "cameraParam.h"

using namespace cv;

void main()
{
	cameraParam param("calibTest.xml");
	Mat src = imread("calibTest.jpg");
	Mat dst = param.undistort(src);
	
	imshow("src", src);
	imshow("dst", dst);
	 
	waitKey(0);
}
