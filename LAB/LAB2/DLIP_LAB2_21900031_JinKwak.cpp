/**
* @about LAB2 Dimension measurement with 2D Camera
* @author Jin Kwak / 21900031, Ignacio / 22320052
* @created 2024.04.09
* @modified 2024.04.12
*/
#include <iostream>
#include <opencv.hpp>
#include <string>
//#include "cameraParam.h"

// Variables Declaration
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
cv::Mat Edge;
// This is to warpPerspective
std::vector<cv::Point2f> srcPoints;
std::vector<cv::Point2f> dstPoints;
cv::Mat perspectiveMatrix;
cv::Mat WarpOut;
// Contour Variables
std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;

void undistort(void);
void warpPerspectiveTransform(void);
void showImg(void);
void lineDetection(void);

int main(){
	src = cv::imread("../../Image/LAB2/Parallel5.jpg",1);
	undistort();

	warpPerspectiveTransform();
	cv::warpPerspective(undistortedSrc, WarpOut, perspectiveMatrix, cv::Size(400, 100));

	for(int idx = 0; idx<8; idx++) cv::medianBlur(WarpOut,WarpOut,3);
	cv::Canny(WarpOut,Edge,60,120);
	// lineDetection();

	cv::Mat thresh;
	cv::threshold(Edge, thresh, 50, 255, cv::THRESH_BINARY);

	findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::cvtColor(WarpOut, WarpOut,cv::COLOR_GRAY2BGR);
	drawContours(WarpOut, contours, -1, cv::Scalar(255,0,255), 1);

	showImg();
	cv::waitKey(0);
	return 0;
}

void undistort(void){
	// Camera Calibration (Undistort)
	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
	cameraMatrix.at<double>(0, 0) = fx;
	cameraMatrix.at<double>(0, 2) = cx;
	cameraMatrix.at<double>(1, 1) = fy;
	cameraMatrix.at<double>(1, 2) = cy;
	distCoeffs.at<double>(0, 0)   = k1;
	distCoeffs.at<double>(1, 0)   = k2;
	distCoeffs.at<double>(2, 0)   = p1;
	distCoeffs.at<double>(3, 0)   = p2;
	cv::undistort(src, undistortedSrc, cameraMatrix, distCoeffs);
	cv::cvtColor(undistortedSrc,undistortedSrc , cv::COLOR_BGR2GRAY);
}

// This is Parallel3
/*void warpPerspectiveTransform(void){
	srcPoints.push_back(cv::Point2f(836, 1620));     // top-left		836, 1620
	srcPoints.push_back(cv::Point2f(3249, 1595 ));    // top-right		3249, 1595
	srcPoints.push_back(cv::Point2f(1013, 1911));     // bottom-left		1013, 1911
	srcPoints.push_back(cv::Point2f(3115, 1911));     // bottom-right	3115, 1911

	dstPoints.push_back(cv::Point2f(0, 0     ));      // top-left
	dstPoints.push_back(cv::Point2f(400, 0   ));     // top-right
	dstPoints.push_back(cv::Point2f(0, 100   ));     // bottom-left
	dstPoints.push_back(cv::Point2f(400, 100 ));    // bottom-right
	perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

	std::cout<<perspectiveMatrix<<std::endl;
}*/

// Parallel 5
void warpPerspectiveTransform(void){
	srcPoints.push_back(cv::Point2f(1003, 1896 ));    // top-left		836, 1620
	srcPoints.push_back(cv::Point2f(3244, 1969 ));    // top-right		3249, 1595
	srcPoints.push_back(cv::Point2f(1147, 2135 ));    // bottom-left	1013, 1911
	srcPoints.push_back(cv::Point2f(3121, 2194 ));    // bottom-right	3115, 1911

	dstPoints.push_back(cv::Point2f(0, 0     ));      // top-left
	dstPoints.push_back(cv::Point2f(400, 0   ));      // top-right
	dstPoints.push_back(cv::Point2f(0, 100   ));      // bottom-left
	dstPoints.push_back(cv::Point2f(400, 100 ));      // bottom-right
	perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

	std::cout<<perspectiveMatrix<<std::endl;
}
void lineDetection(){
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(Edge, lines, 1, CV_PI / 180, 120, 0, 0);
	// Draw the detected lines
	for (cv::Vec2f line: lines){
		float rho = line[0], theta = line[1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cv::line(WarpOut, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}
}

void showImg(){
	cv::namedWindow("Undistort",cv::WINDOW_GUI_NORMAL);
	cv::imshow("Undistort", undistortedSrc);
	cv::namedWindow("Warped", cv::WINDOW_GUI_NORMAL);
	cv::imshow("Warped",WarpOut);
	cv::namedWindow("Edge",cv::WINDOW_GUI_NORMAL);
	cv::imshow("Edge", Edge);
}