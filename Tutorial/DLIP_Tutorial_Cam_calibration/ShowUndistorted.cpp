#include <iostream>
#include <opencv.hpp>



int main()
{
	cv::Mat src, result, cameraMatrix, distCoeffs;
	src = cv::imread("D:\\DLIP\\Tutorial\\DLIP_Tutorial_Cam_calibration\\calibTest.jpg");
 
	double fx, fy, cx, cy, k1, k2, p1, p2;
    
	fx = 834.588;            // Focal length
	fy = 847.428;            // Focal length
	cx = 593.712;
	cy = 355.082;
	k1 = -0.407936;
	k2 = 0.141814;
	p1 = -0.001916;
	p2 = 0.0007233;

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
 