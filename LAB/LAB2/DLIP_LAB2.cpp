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
#define CORNER_COEFF     (double)(0.002)
#define MAX_CLUSTER      (const int)(17)
#define NEXT_RECT        (int)(4)

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

// Cluster Variables
std::vector<cv::Point2f> Corner_points_Front;
std::vector<cv::Point2f> Corner_points_Up;
cv::Mat Cluster_Center_Front;
cv::Mat Cluster_Center_Up;

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

void onMouse(int event, int x, int y, int flags, void* userdata) {
	PointData* pd = (PointData*)userdata;
	if (event == cv::EVENT_LBUTTONDOWN) pd->points.push_back(cv::Point(x, y));
}

void undistort(void);
void warpPerspectiveTransform(void);
void showImg(void);
void cluster(void);
void sortPointsX(cv::Mat& clusterCenters);
void sortPointsY(cv::Mat& clusterCenters);
void removeRow(cv::Mat& original, int rowIndex);
void sortRectangleCorners(cv::Mat& points);
void printRows(cv::Mat& mat);
void getPix2mm(void);
void getLength(void);
PointData pointData_Up;
PointData pointData_Front;

int main(){
	src_upper_upper = cv::imread("../../Image/LAB2/Corner.jpg");
	src_front_front = cv::imread("../../Image/LAB2/Front.jpg");
	undistort();
	cv::namedWindow	("Click 4 points of Upper Image",cv::WINDOW_GUI_NORMAL);
	cv::imshow		("Click 4 points of Upper Image", undistortedsrc_upper);
	cv::setMouseCallback("Click 4 points of Upper Image", onMouse, &pointData_Up);
	cv::waitKey		(0);
	cv::namedWindow	("Click 4 points of Front Image",cv::WINDOW_GUI_NORMAL);
	cv::imshow		("Click 4 points of Front Image", undistortedsrc_front);
	cv::setMouseCallback("Click 4 points of Front Image", onMouse, &pointData_Front);
	cv::waitKey		(0);
	warpPerspectiveTransform();
	cv::imshow("Warped Image",WarpOut);
	cv::waitKey(0);
 	// Filter
	for(int idx = 0; idx<20; idx++) cv::medianBlur(WarpOut,WarpOut,3);
	cv::cornerHarris(WarpOut,Corner_Upper,BLOCK_SIZE,APERTURE_SIZE,CORNER_COEFF);
	Corner_Upper*=255;
	//Clustering algorithm
	cluster();

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

//Corner
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
	cv::imshow("Corner", Corner_Upper);
}

void sortPointsY(cv::Mat& clusterCenters) {
	std::vector<cv::Point2f> points;

	// Extract points from the Mat
	for (int i = 0; i < clusterCenters.rows; i++) {
		float x = clusterCenters.at<float>(i, 0);
		float y = clusterCenters.at<float>(i, 1);
		points.push_back(cv::Point2f(x, y));
	}
	// Sort points by x-coordinate
	std::sort(points.begin(), points.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
	    return a.y < b.y || (a.y == b.y && a.x < b.x);  // Secondary sort by y if x is the same
	});
	// Optionally, put the sorted points back into the matrix if needed
	for (int i = 0; i < clusterCenters.rows; i++) {
		clusterCenters.at<float>(i, 0) = points[i].x;
		clusterCenters.at<float>(i, 1) = points[i].y;
	}
}

void sortPointsX(cv::Mat& clusterCenters) {
	std::vector<cv::Point2f> points;

	// Extract points from the Mat
	for (int i = 0; i < clusterCenters.rows; i++) {
		float x = clusterCenters.at<float>(i, 0);
		float y = clusterCenters.at<float>(i, 1);
		points.push_back(cv::Point2f(x, y));
	}
	// Sort points by x-coordinate
	std::sort(points.begin(), points.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
	    return a.x < b.x || (a.x == b.x && a.y < b.y);  // Secondary sort by y if x is the same
	});
	// Optionally, put the sorted points back into the matrix if needed
	for (int i = 0; i < clusterCenters.rows; i++) {
		clusterCenters.at<float>(i, 0) = points[i].x;
		clusterCenters.at<float>(i, 1) = points[i].y;
	}
}

void removeRow(cv::Mat& original, int rowIndex) {
	// Check if the row index is valid
	if (rowIndex < 0 || rowIndex >= original.rows) {
		std::cout << "Row index out of bounds" << std::endl;
	}
	else{
		// Create a new matrix with one row less
		cv::Mat reduced(original.rows - 1, original.cols, original.type());

		// Copy the data from the original matrix to the new one, skipping the specified row
		for (int i = 0, j = 0; i < original.rows; i++)	if (i != rowIndex)	original.row(i).copyTo(reduced.row(j++));
		// Replace the original matrix with the new one
		original = reduced;
	}
}

void sortRectangleCorners(cv::Mat& points) {
	// Iterate through each set of four points
	for (int i = 0; i < points.rows; i += 4) {
		// Ensure we don't run out of bounds
		if (i + 3 >= points.rows) break;

		// Extract top two points into a vector and sort by x
		std::vector<cv::Point2f> topCorners = {points.at<cv::Point2f>(i, 0), points.at<cv::Point2f>(i + 1, 0)};
		std::sort(topCorners.begin(), topCorners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
		    return a.x < b.x;
		});

		// Assign sorted top corners back to the Mat
		points.at<cv::Point2f>(i, 0) = topCorners[0];
		points.at<cv::Point2f>(i + 1, 0) = topCorners[1];

		// Extract bottom two points into a vector and sort by x
		std::vector<cv::Point2f> bottomCorners = {points.at<cv::Point2f>(i + 2, 0), points.at<cv::Point2f>(i + 3, 0)};
		std::sort(bottomCorners.begin(), bottomCorners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
		    return a.x < b.x;
		});
		// Assign sorted bottom corners back to the Mat
		points.at<cv::Point2f>(i + 2, 0) = bottomCorners[0];
		points.at<cv::Point2f>(i + 3, 0) = bottomCorners[1];
	}
}

//Print Cluster Points
void printRows(cv::Mat& mat){
	for(int idx = 0; idx<mat.rows; idx++) {
		std::cout<<mat.at<float>(idx,0)<<" , "<<mat.at<float>(idx,1)<<std::endl;
	}
}

void getPix2mm(void){
	float bigTRx = Cluster_Center_Up.at<float>(TRIGHT,0);
	float bigTRy = Cluster_Center_Up.at<float>(TRIGHT,1);
	float bigTLx = Cluster_Center_Up.at<float>(TLEFT,0);
	float bigTLy = Cluster_Center_Up.at<float>(TLEFT,1);
	float bigBLx = Cluster_Center_Up.at<float>(NEXT_RECT+BLEFT,0);
	float bigBLy = Cluster_Center_Up.at<float>(NEXT_RECT+BLEFT,1);
	float bigBRx = Cluster_Center_Up.at<float>(NEXT_RECT+BRIGHT,0);
	float bigBRy = Cluster_Center_Up.at<float>(NEXT_RECT+BRIGHT,1);
	float smallTLx = Cluster_Center_Up.at<float>(NEXT_RECT*2+TLEFT,0);
	float smallTLy = Cluster_Center_Up.at<float>(NEXT_RECT*2+TLEFT,1);
	float smallBRx = Cluster_Center_Up.at<float>(NEXT_RECT*3+BRIGHT,0);
	float smallBRy = Cluster_Center_Up.at<float>(NEXT_RECT*3+BRIGHT,1);

	//Check If the edge is correctly selected
	cv::circle(WarpOut,cv::Point2f(bigTRx,bigTRy),3, 255);
	cv::circle(WarpOut,cv::Point2f(bigTLx,bigTLy),3, 255);
	cv::circle(WarpOut,cv::Point2f(bigBRx,bigBRy),3, 255);
	cv::circle(WarpOut,cv::Point2f(bigBLx,bigBLy),3, 255);
	cv::circle(WarpOut,cv::Point2f(smallTLx,smallTLy),3, 255);
	cv::circle(WarpOut,cv::Point2f(smallBRx,smallBRy),3, 255);
	// Pixel To mm
	PIXEL2MM_X = 50/fabs(smallTLx- smallBRx);
	PIXEL2MM_Y = 50/fabs(smallTLy- smallBRy);
	std::cout<<"Pixel to mm conversion (X-axis)  = "<<PIXEL2MM_X<<std::endl;
	std::cout<<"Pixel to mm conversion (Y-axis)  = "<<PIXEL2MM_Y<<std::endl;
	Volume[CAM_X] = ((fabs(bigTRx- bigBLx)+fabs(Cluster_Center_Up.at<float>(BRIGHT,0) - Cluster_Center_Up.at<float>(NEXT_RECT+TLEFT,0)))/2)*PIXEL2MM_X;
	Volume[CAM_Y] = ((fabs(bigTRy- bigBRy)+fabs(bigTRy-bigBLy)+fabs(bigTLy- bigBLy)+fabs(bigTLy-bigBRy))/4)*PIXEL2MM_Y;
}

void cluster(){
	cv::Mat Cluster_bestLabel_Front;
	cv::Mat Cluster_bestLabel_Up;
	for(int row= 0; row<Corner_Upper.rows; row++)
		for(int col = 0; col<Corner_Upper.cols; col++)
			if(Corner_Upper.at<float>(row,col)>=.1f)	Corner_points_Up.push_back(cv::Point2f(col,row));

	cv::Mat data(Corner_points_Up.size(),2,CV_32F);
	for(size_t idx = 0; idx<Corner_points_Up.size();idx++) {
		data.at<float>(idx,U) = Corner_points_Up[idx].x;
		data.at<float>(idx,V) = Corner_points_Up[idx].y;
	 }
	cv::kmeans(data,MAX_CLUSTER, Cluster_bestLabel_Up,
			  cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1),
			  10,cv::KMEANS_PP_CENTERS,Cluster_Center_Up);
	sortPointsY(Cluster_Center_Up);

	removeRow(Cluster_Center_Up,6);
	sortRectangleCorners(Cluster_Center_Up);
}