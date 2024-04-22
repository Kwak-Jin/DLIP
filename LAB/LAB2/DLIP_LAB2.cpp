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

enum CAMERA_FRAME{
	CAM_X   = 0,
	CAM_Y   = 1,
	CAM_Z   = 2,
	N_CAM   = 3,
};
enum PIXEL_FRAME{
	U       = 0,
	V       = 1,
	N_PIX   = 2
};

enum CORNER{
	TLEFT    = 0,
	TRIGHT   = 1,
	BLEFT    = 2,
	BRIGHT   = 3,
	N_CORNER = 4
};

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
	std::cout<<"Volume = "<<Size<<"mm^3"<<std::endl;
	return Size;
};
void undistort(void);
void warpPerspectiveTransform(void);
void showImg(void);
void cluster(void);
void sortPointsX(cv::Mat& clusterCenters);
void sortPointsY(cv::Mat& clusterCenters);
void removeRow(cv::Mat& original, int rowIndex);
void sortRectangleCorners(cv::Mat& points);
void assignCentroids_Up(void);
void printRows(cv::Mat& mat);
void getDiff_Up(void);
void getLength(void);

int main(){
	src_upper_upper = cv::imread("../../Image/LAB2/Corner.jpg");
	src_front_front = cv::imread("../../Image/LAB2/Front.jpg");
	undistort();
	warpPerspectiveTransform();
 	// Filter
	for(int idx = 0; idx<10; idx++) cv::medianBlur(WarpOut,WarpOut,3);
	cv::cornerHarris(WarpOut,Corner_Upper,BLOCK_SIZE,APERTURE_SIZE,CORNER_COEFF);
	Corner_Upper*=255;
	//Clustering algorithm
	cluster();

	getDiff_Up();
	Volume[CAM_Z] = 50.f;
	float Vol = GetVolume(Volume);
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
	src_Upper.push_back(cv::Point2f(1198, 1043));    //967, 979
	src_Upper.push_back(cv::Point2f(2112, 1039));    //3087, 977
	src_Upper.push_back(cv::Point2f(1189, 2874));    //967, 1523)
	src_Upper.push_back(cv::Point2f(2100, 2924));    //3076, 1523

	dstPoints_Upper.push_back(cv::Point2f(0, 0      ));          // top-left
	dstPoints_Upper.push_back(cv::Point2f(200, 0    ));          // top-right
	dstPoints_Upper.push_back(cv::Point2f(0, 800    ));          // bottom-left
	dstPoints_Upper.push_back(cv::Point2f(200, 800  ));          // bottom-right
	perspectiveMatrix = cv::getPerspectiveTransform(src_Upper, dstPoints_Upper);
	cv::warpPerspective(undistortedsrc_upper, WarpOut, perspectiveMatrix, cv::Size(200, 800));
	std::cout<<perspectiveMatrix<<std::endl;
	//Front
	src_Front.push_back(cv::Point2f(665, 967));    //967, 979
	src_Front.push_back(cv::Point2f(3455, 895));    //3087, 977
	src_Front.push_back(cv::Point2f(721, 1651));    //967, 1523)
	src_Front.push_back(cv::Point2f(3422, 1596));    //3076, 1523

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
		for (int i = 0, j = 0; i < original.rows; i++) {
			if (i != rowIndex) {
				original.row(i).copyTo(reduced.row(j++));
			}
		}
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

void assignCentroids_Up(void){
	//Initial Centroids
	Cluster_Center_Up = cv::Mat(MAX_CLUSTER,2,CV_32F);
	Cluster_Center_Up.at<float>(0 ,U)  = 166;
	Cluster_Center_Up.at<float>(1 ,U)  = 200;
	Cluster_Center_Up.at<float>(2 ,U)  = 166;
	Cluster_Center_Up.at<float>(3 ,U)  = 200;
	Cluster_Center_Up.at<float>(4 ,U)  = 0  ;
	Cluster_Center_Up.at<float>(5 ,U)  = 50 ;
	Cluster_Center_Up.at<float>(6 ,U)  = 0  ;
	Cluster_Center_Up.at<float>(7 ,U)  = 50 ;
	Cluster_Center_Up.at<float>(8 ,U)  = 0  ;
	Cluster_Center_Up.at<float>(9 ,U)  = 50 ;
	Cluster_Center_Up.at<float>(10,U)  = 0  ;
	Cluster_Center_Up.at<float>(11,U)  = 50 ;
	Cluster_Center_Up.at<float>(12,U)  = 60 ;
	Cluster_Center_Up.at<float>(13,U)  = 110;
	Cluster_Center_Up.at<float>(14,U)  = 60 ;
	Cluster_Center_Up.at<float>(15,U)  = 150;
	Cluster_Center_Up.at<float>(0 ,V) =  0 ;
	Cluster_Center_Up.at<float>(1 ,V) =  0 ;
	Cluster_Center_Up.at<float>(2 ,V) =  50*2;
	Cluster_Center_Up.at<float>(3 ,V) =  50*2;
	Cluster_Center_Up.at<float>(4 ,V) =  150*2;
	Cluster_Center_Up.at<float>(5 ,V) =  150*2;
	Cluster_Center_Up.at<float>(6 ,V) =  200*2;
	Cluster_Center_Up.at<float>(7 ,V) =  200*2;
	Cluster_Center_Up.at<float>(8 ,V) =  300*2;
	Cluster_Center_Up.at<float>(9 ,V) =  300*2;
	Cluster_Center_Up.at<float>(10,V) =  330*2;
	Cluster_Center_Up.at<float>(11,V) =  330*2;
	Cluster_Center_Up.at<float>(12,V) =  360*2;
	Cluster_Center_Up.at<float>(13,V) =  360*2;
	Cluster_Center_Up.at<float>(14,V) =  395*2;
	Cluster_Center_Up.at<float>(15,V) =  395*2;
}

//Print Cluster Points
void printRows(cv::Mat& mat){
	for(int idx = 0; idx<mat.rows; idx++) {
		std::cout<<mat.at<float>(idx,0)<<" , "<<mat.at<float>(idx,1)<<std::endl;
	}
}

void getDiff_Up(void){
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
	assignCentroids_Up();
	cv::kmeans(data,MAX_CLUSTER, Cluster_bestLabel_Up,
			  cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1),
			  10,cv::KMEANS_PP_CENTERS,Cluster_Center_Up);
	sortPointsY(Cluster_Center_Up);

	removeRow(Cluster_Center_Up,6);
	sortRectangleCorners(Cluster_Center_Up);
}