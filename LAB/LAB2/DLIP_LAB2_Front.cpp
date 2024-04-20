/**
* @about LAB2 Dimension measurement with 2D Camera
* @author Jin Kwak / 21900031, Ignacio / 22320052
* @created 2024.04.09
* @modified 2024.04.12
*/
#include <iostream>
#include <opencv.hpp>
#include <string>
#define blockSize		(int)(2)
#define apertureSize	(int)(3)

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
cv::Mat thresh;

// Contour Variables
std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;

void undistort(void);
void warpPerspectiveTransform(void);
void showImg(void);

int main(){
	src = cv::imread("../../Image/LAB2/Front.jpg");
	undistort();
	warpPerspectiveTransform();
	// Filter
	for(int idx = 0; idx<10; idx++) cv::medianBlur(WarpOut,WarpOut,3);
	cv::cornerHarris(WarpOut,Edge,blockSize,apertureSize,0.001);
	// cv::convertScaleAbs(Edge,Edge);
	// cv::equalizeHist(Edge,Edge);
	Edge*=255;
	std::cout<<"MaxRow ="<< Edge.rows<<"Maxcol ="<<Edge.cols<<std::endl;
	for(int row= 0; row<Edge.rows; row++)
		for(int col = 0; col<Edge.cols; col++)
			if(Edge.at<float>(row,col)>=.1f) std::cout<<row <<" "<< col<<std::endl;

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

//Corner
void warpPerspectiveTransform(void){
	srcPoints.push_back(cv::Point2f(665, 967));    //967, 979
	srcPoints.push_back(cv::Point2f(3455, 895));    //3087, 977
	srcPoints.push_back(cv::Point2f(721, 1651));    //967, 1523)
	srcPoints.push_back(cv::Point2f(3422, 1596));    //3076, 1523

	dstPoints.push_back(cv::Point2f(0, 0      ));    // top-left
	dstPoints.push_back(cv::Point2f(800, 0    ));    // top-right
	dstPoints.push_back(cv::Point2f(0, 200    ));    // bottom-left
	dstPoints.push_back(cv::Point2f(800, 200  ));    // bottom-right
	perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
	cv::warpPerspective(undistortedSrc, WarpOut, perspectiveMatrix, cv::Size(800, 200));

	std::cout<<perspectiveMatrix<<std::endl;
}


void showImg(){
	cv::namedWindow("Undistort",cv::WINDOW_GUI_NORMAL);
	cv::imshow("Undistort", undistortedSrc);
	cv::namedWindow("Warped", cv::WINDOW_AUTOSIZE);
	cv::imshow("Warped",WarpOut);
	cv::imshow("Corner", Edge);
}