/**
* @about LAB2 Dimension measurement with 2D Camera (Mouse Click to extract Pixel indices)
* @author Jin Kwak / 21900031, Ignacio / 22320052
* @created 2024.04.09
* @modified 2024.04.12
*/
#include <iostream>
#include <opencv.hpp>
#include <string>

// Camera Calibration constants
const double fx = 3040.3677120589309   ;
const double fy= 3045.2493629364403    ;
const double cx= 2025.5142669069799    ;
const double cy= 1505.5083838246874    ;
const double k1= 0.077964106090950946  ;
const double k2= -0.1524773352777998   ;
const double p1= 0.0018974244228679193 ;
const double p2= -0.002899002140939538 ;
cv::Mat cameraMatrix, distCoeffs;

cv::Mat src				  ;
cv::Mat undistortedSrc	  ;


// Structure to store clicked points
struct PointData {
	std::vector<cv::Point> points;
};

PointData* onMouse(int event, int x, int y, int flags, void* userdata) {
	PointData* pd = (PointData*)userdata;
	if (event == cv::EVENT_LBUTTONDOWN) {
		pd->points.push_back(cv::Point(x, y));
		std::cout << "Clicked at: " << x << ", " << y << std::endl;
	}
	else if(event == cv::EVENT_RBUTTONDOWN) {
	}
	return pd;
}

int main(){
	// Camera Calibration (Undistort)

	src = cv::imread("../../Image/LAB2/Front.jpg",1);

	PointData pointData;

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

	cv::undistort(src, undistortedSrc, cameraMatrix, distCoeffs);

	cv::namedWindow("Undistort",cv::WINDOW_GUI_NORMAL);
	cv::imshow("Undistort", undistortedSrc);
	cv::setMouseCallback("Undistort", onMouse, &pointData);

	cv::waitKey(0);
	std::cout << "Clicked Points:" << std::endl;
	for (const auto& point : pointData.points) {
		std::cout << point << std::endl;
	}
	return 0;
}