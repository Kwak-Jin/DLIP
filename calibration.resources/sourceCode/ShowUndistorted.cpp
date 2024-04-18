#include <iostream>
#include <opencv.hpp>



int main()
{
	cv::Mat src, result, cameraMatrix, distCoeffs;
	src = cv::imread("test.jpg");
 
	double fx, fy, cx, cy, k1, k2, p1, p2;
    
	fx = 818.51;
	fy = 826.80;
	cx = 651.75;
	cy = 332.74;
	k1 = -0.4458;
	k2 = 0.2294; 
	p1 = 0.0036;
	p2 = -0.0009;
	
	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = fx;
	cameraMatrix.at<double>(0, 2) = cx;
	cameraMatrix.at<double>(1, 1) = fy;
	cameraMatrix.at<double>(1, 2) = cy;


	distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
	distCoeffs.at<double>(0, 0) = k1;
	distCoeffs.at<double>(1, 0) = k2;
	distCoeffs.at<double>(2, 0) = p1;
	distCoeffs.at<double>(3, 0) = p2;



	cv::undistort(src, result, cameraMatrix, distCoeffs);



	cv::imshow("SRC",	src);
	cv::imshow("result", result);
	cv::waitKey(0);
	return 0;
}
 