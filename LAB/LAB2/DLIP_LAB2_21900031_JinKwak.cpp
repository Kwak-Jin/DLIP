/** @about LAB2 Dimension measurement with 2D Camera
*   @author Jin Kwak / 21900031, Ignacio / 22320052
*   @created 2024.04.09
*   @modified 2024.04.23
*/

#include <iostream>
#include <opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#define BLOCK_SIZE		(int)(2)
#define APERTURE_SIZE	(int)(3)

#define NEXT_RECT        (int)(4)
#define N_IDX            (int)(50)
#define K_SIZE           (int)(3)
enum CAMERA_FRAME{ CAM_X   = 0, CAM_Y   = 1, CAM_Z   = 2, N_CAM   = 3};
enum PIXEL_FRAME{  U       = 0, V       = 1,	N_PIX   = 2};
enum CORNER{ TLEFT = 0,	TRIGHT = 1, BLEFT = 2, BRIGHT = 3,	N_CORNER = 4};

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

cv::Mat src_upper_upper				  ;
cv::Mat undistortedsrc_upper			  ;
cv::Mat Corner_Upper				  ;

cv::Mat src_front_front				  ;
cv::Mat undistortedsrc_front			  ;
cv::Mat Corner_front				  ;

// This is to warpPerspective
std::vector<cv::Point2f> src_Upper	       ;
std::vector<cv::Point2f> dstPoints_Upper  ;
cv::Mat perspectiveMatrix			  ;
cv::Mat WarpOut				       ;

std::vector<cv::Point2f> src_Front	       ;
std::vector<cv::Point2f> dstPoints_Front  ;
cv::Mat perspectiveMatrix_F			  ;
cv::Mat WarpOut_F				       ;

float PIXEL2MM_X;
float PIXEL2MM_Y;
float Volume[N_CAM] = {0.f,};
inline float GetVolume(float* Vol){
	float Size = 1;
	for(int idx= 0; idx<N_CAM; idx++) {
		std::cout<<"Volume Parameter("<<idx<<") = "<<Vol[idx]<<"[mm]"<<std::endl;
		Size*= Vol[idx];
	}
	return Size;
};

// Structure to store clicked points
struct PointData {
	std::vector<cv::Point2f> points;
};

std::vector<cv::Point> points;
int draggingPoint = -1;   // Index of the dragging point, -1 if no point is being dragged
const int radius = 5;     // Radius for drawing points
const int thickness = -1; // Fill the circle

void onMouse(int event, int x, int y, int flags, void* userdata) {
	PointData* pd = (PointData*)userdata;
	if (event == cv::EVENT_LBUTTONDOWN) pd->points.push_back(cv::Point(x, y));
}


void undistort(void);
void warpPerspectiveTransform(void);
void showImg(void);
void getPix2mm(void);
void getLength(void);
PointData pointData_Up;
PointData pointData_Front;

int main(){
	src_upper_upper = cv::imread("../../Image/LAB2/Corner.jpg");
	src_front_front = cv::imread("../../Image/LAB2/Front.jpg");
	undistort();
	std::cout<<"Click 4 points of the edge"<<std::endl;
	cv::namedWindow	("Click 4 points of Upper Image",cv::WINDOW_GUI_NORMAL);
	cv::imshow		("Click 4 points of Upper Image", undistortedsrc_upper);
	cv::setMouseCallback("Click 4 points of Upper Image", onMouse, &pointData_Up);
	cv::waitKey		(0);
	std::cout<<"Click 4 points of the edge"<<std::endl;
	cv::namedWindow	("Click 4 points of Front Image",cv::WINDOW_GUI_NORMAL);
	cv::imshow		("Click 4 points of Front Image", undistortedsrc_front);
	cv::setMouseCallback("Click 4 points of Front Image", onMouse, &pointData_Front);
	cv::waitKey		(0);
	warpPerspectiveTransform();
 	// Filter
	for(int idx = 0; idx<N_IDX; idx++) {
		cv::medianBlur(WarpOut,WarpOut,K_SIZE);
		cv::medianBlur(WarpOut,WarpOut_F,K_SIZE);
	}

	getPix2mm();
	Volume[CAM_Z] = 50.f;
	float Vol = GetVolume(Volume);
	std::cout<<"Volume = "<<Vol<<" mm^3"<<std::endl;

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
	cv::undistort(src_upper_upper, undistortedsrc_upper, cameraMatrix, distCoeffs);
	cv::cvtColor(undistortedsrc_upper,undistortedsrc_upper , cv::COLOR_BGR2GRAY);
	cv::undistort(src_front_front,undistortedsrc_front, cameraMatrix,distCoeffs);
	cv::cvtColor(undistortedsrc_front,undistortedsrc_front, cv::COLOR_BGR2GRAY);
}

void warpPerspectiveTransform(void){
	for (const auto& point : pointData_Up.points) src_Upper.push_back(point); //1198, 1043 //2112, 1039 //1189, 2874  //2100, 2924
	//assign the Size of  output matrix
	dstPoints_Upper.push_back(cv::Point2f(0, 0      ));          // top-left
	dstPoints_Upper.push_back(cv::Point2f(200, 0    ));          // top-right
	dstPoints_Upper.push_back(cv::Point2f(0, 800    ));          // bottom-left
	dstPoints_Upper.push_back(cv::Point2f(200, 800  ));          // bottom-right
	perspectiveMatrix = cv::getPerspectiveTransform(src_Upper, dstPoints_Upper);
	cv::warpPerspective(undistortedsrc_upper, WarpOut, perspectiveMatrix, cv::Size(200, 800));

	//Front
	for (const auto& point : pointData_Front.points) src_Front.push_back(point); //665, 967  //3455, 895 //721, 1651 //3422, 1596
	dstPoints_Front.push_back(cv::Point2f(0, 0      ));    // top-left
	dstPoints_Front.push_back(cv::Point2f(800, 0    ));    // top-right
	dstPoints_Front.push_back(cv::Point2f(0, 200    ));    // bottom-left
	dstPoints_Front.push_back(cv::Point2f(800, 200  ));    // bottom-right
	perspectiveMatrix = cv::getPerspectiveTransform(src_Front, dstPoints_Front);
	cv::warpPerspective(undistortedsrc_front, WarpOut_F, perspectiveMatrix, cv::Size(800, 200));
}

void showImg(){
	cv::namedWindow("Undistort",cv::WINDOW_GUI_NORMAL);
	cv::imshow("Undistort", undistortedsrc_upper);
	cv::namedWindow("Warped", cv::WINDOW_AUTOSIZE);
	cv::imshow("Warped",WarpOut);
	cv::imshow("Warped Front",WarpOut_F);
}

void getPix2mm(void){
	float bigTRx   = 0.0;
	float bigTRy   = 0.0;
	float bigTLx   = 0.0;
	float bigTLy   = 0.0;
	float bigBLx   = 0.0;
	float bigBLy   = 0.0;
	float bigBRx   = 0.0;
	float bigBRy   = 0.0;
	float smallTLx = 0.0;
	float smallTLy = 0.0;
	float smallBRx = 0.0;
	float smallBRy = 0.0;

	// Pixel To mm
	PIXEL2MM_X = 0;//50/fabs(smallTLx- smallBRx);
	PIXEL2MM_Y = 0;//50/fabs(smallTLy- smallBRy);
	std::cout<<"Pixel to mm conversion (X-axis)  = "<<PIXEL2MM_X<<std::endl;
	std::cout<<"Pixel to mm conversion (Y-axis)  = "<<PIXEL2MM_Y<<std::endl;
	Volume[CAM_X] = PIXEL2MM_X;
	Volume[CAM_Y] = PIXEL2MM_Y;
}

