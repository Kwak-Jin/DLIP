/** @about LAB2 Dimension measurement with 2D Camera
*   @author Jin Kwak / 21900031, Ignacio / 22320052
*   @created 2024.04.09
*   @modified 2024.04.30
*/

#include <iostream>
#include <opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#define CORNER_COEFF     (double)(0.002)
#define MAX_CLUSTER      (const int)(16)
#define NEXT_RECT        (int)(4)
#define N_IDX            (int)(50)
enum CAMERA_FRAME{ CAM_X   = 0, CAM_Y   = 1, CAM_Z   = 2, N_CAM   = 3};
enum PIXEL_FRAME{  U       = 0, V       = 1};
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

cv::Mat cameraMatrix, distCoeffs		;
cv::Mat src_upper_upper				;
cv::Mat undistortedsrc_upper			;
cv::Mat Corner_Upper				;

cv::Mat src_front_front				;
cv::Mat undistortedsrc_front			;
cv::Mat Corner_front				;

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
float PIXEL2MM_Z;
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
int draggingPoint = -1;  // Index of the dragging point, -1 if no point is being dragged
const int radius = 5;    // Radius for drawing points
const int thickness = -1; // Fill the circle

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
void sortRectangleCorners(cv::Mat& points);
void getPix2mm(void);
PointData pointData_Up;
PointData pointData_Front;

int main(){
	src_upper_upper = cv::imread("../../Image/LAB2/Up.jpg");
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
		cv::medianBlur(WarpOut,WarpOut,3);
		cv::medianBlur(WarpOut_F,WarpOut_F,3);
	}
	for(int idx = 0; idx<3; idx++) {
		cv::medianBlur(WarpOut,WarpOut,7);
		cv::medianBlur(WarpOut_F,WarpOut_F,7);
	}
	//Corner Detection
	cv::cornerHarris(WarpOut,Corner_Upper,2,3,CORNER_COEFF);
	cv::cornerHarris(WarpOut_F,Corner_front,2,3,CORNER_COEFF);

	//Clustering algorithm
	cluster();
	getPix2mm();
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

void getPix2mm(void){
	// Pixel To mm
	PIXEL2MM_X = 50.0/fabs(Cluster_Center_Up.at<float>(NEXT_RECT*2+TLEFT,0)- Cluster_Center_Up.at<float>(NEXT_RECT*3+BRIGHT,0));
	PIXEL2MM_Y = 50.0/fabs(Cluster_Center_Up.at<float>(NEXT_RECT*2+TLEFT,1)- Cluster_Center_Up.at<float>(NEXT_RECT*3+BRIGHT,1));
	PIXEL2MM_Z = 50.0/fabs(Cluster_Center_Front.at<float>(NEXT_RECT+TRIGHT,0) - Cluster_Center_Up.at<float>(BLEFT,0));
	std::cout<<"Pixel to mm conversion (X-axis)  = "<<PIXEL2MM_X<<std::endl;
	std::cout<<"Pixel to mm conversion (Y-axis)  = "<<PIXEL2MM_Y<<std::endl;
	std::cout<<"Pixel to mm conversion (Z-axis)  = "<<PIXEL2MM_Z<<std::endl;

	Volume[CAM_X] = ((fabs(Cluster_Center_Up.at<float>(TRIGHT,0)- Cluster_Center_Up.at<float>(NEXT_RECT+BLEFT,0))))*PIXEL2MM_X;
	Volume[CAM_Y] = ((fabs(Cluster_Center_Up.at<float>(TRIGHT,0)- Cluster_Center_Up.at<float>(NEXT_RECT+BRIGHT,1))+fabs(Cluster_Center_Up.at<float>(TLEFT,1)-Cluster_Center_Up.at<float>(NEXT_RECT+BRIGHT,1)))/2)*PIXEL2MM_Y;
	Volume[CAM_Z] = fabs(Cluster_Center_Front.at<float>(3*NEXT_RECT+TRIGHT,1) - Cluster_Center_Front.at<float>(2*NEXT_RECT+BLEFT,1))*PIXEL2MM_Z;
}

void cluster(){
	cv::Mat Cluster_bestLabel_Front;
	cv::Mat Cluster_bestLabel_Up;
	// Amplification
	Corner_Upper*=255;
	Corner_front*=255;
	for(int row= 0; row<Corner_Upper.rows; row++)
		for(int col = 0; col<Corner_Upper.cols; col++)
			if(Corner_Upper.at<float>(row,col)>=.09f) {
				Corner_points_Up.push_back(cv::Point2f(col,row));
				//std::cout<<col <<","<< row <<std::endl;
			}
	for(int row= 0; row<Corner_front.rows; row++)
		for(int col = 0; col<Corner_front.cols; col++)
			if(Corner_front.at<float>(row,col)>=.1f) {
				Corner_points_Front.push_back(cv::Point2f(col,row));
			}
	cv::Mat data_Up(Corner_points_Up.size(),2,CV_32F);
	cv::Mat data_Front(Corner_points_Front.size(),2,CV_32F);
	for(size_t idx = 0; idx<Corner_points_Up.size();idx++) {
		data_Up.at<float>(idx,U) = Corner_points_Up[idx].x;
		data_Up.at<float>(idx,V) = Corner_points_Up[idx].y;
	}
	for(size_t idx = 0; idx<Corner_points_Front.size();idx++) {
		data_Front.at<float>(idx,U) = Corner_points_Front[idx].x;
		data_Front.at<float>(idx,V) = Corner_points_Front[idx].y;
	}
	cv::kmeans(data_Up,MAX_CLUSTER, Cluster_bestLabel_Up,
			  cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1),
			  10,cv::KMEANS_PP_CENTERS,Cluster_Center_Up);
	sortPointsY(Cluster_Center_Up);

	cv::kmeans(data_Front,MAX_CLUSTER, Cluster_bestLabel_Front,
			  cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1),
			  10,cv::KMEANS_PP_CENTERS,Cluster_Center_Front);
	sortPointsX(Cluster_Center_Front);
	sortRectangleCorners(Cluster_Center_Up);
	sortRectangleCorners(Cluster_Center_Front);
}