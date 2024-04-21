/**
* @about LAB2 Dimension measurement with 2D Camera
* @author Jin Kwak / 21900031, Ignacio / 22320052
* @created 2024.04.09
* @modified 2024.04.12
*/
#include <iostream>
#include <opencv.hpp>
#include <string>
#define BLOCK_SIZE		(int)(2)
#define APERTURE_SIZE	(int)(3)
#define CORNER_COEFF     (double)(0.002)
#define MAX_CLUSTER      (const int)(17)


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

cv::Mat src_upper_upper				  ;
cv::Mat undistortedsrc_upper			  ;
cv::Mat Edge						  ;
// This is to warpPerspective
std::vector<cv::Point2f> src_upperPoints  ;
std::vector<cv::Point2f> dstPoints		  ;
cv::Mat perspectiveMatrix			  ;
cv::Mat WarpOut				       ;

// Contour Variables
std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;

// Cluster Variables
std::vector<cv::Point2f> Edge_points_Front;
std::vector<cv::Point2f> Edge_points_Up;
cv::Mat Cluster_Center_Front;
cv::Mat Cluster_Center_Up;
cv::Mat Cluster_bestLabel_Front;
cv::Mat Cluster_bestLabel_Up;


void undistort(void);
void warpPerspectiveTransform(void);
void showImg(void);
int findOutlierCluster(const cv::Mat& centers);
void sortPoints(cv::Mat& clusterCenters);

int main(){
	src_upper_upper = cv::imread("../../Image/LAB2/Corner.jpg");
	undistort();
	warpPerspectiveTransform();
 	// Filter
	for(int idx = 0; idx<10; idx++) cv::medianBlur(WarpOut,WarpOut,3);
	cv::cornerHarris(WarpOut,Edge,BLOCK_SIZE,APERTURE_SIZE,CORNER_COEFF);
	Edge*=255;
	std::cout<<"MaxRow ="<< Edge.rows<<"Maxcol ="<<Edge.cols<<std::endl;
	for(int row= 0; row<Edge.rows; row++)
		for(int col = 0; col<Edge.cols; col++)
			if(Edge.at<float>(row,col)>=.1f)	Edge_points_Up.push_back(cv::Point2f(col,row));

	cv::Mat data(Edge_points_Up.size(),2,CV_32F);
	for(size_t idx = 0; idx<Edge_points_Up.size();idx++) {
		data.at<float>(idx,U) = Edge_points_Up[idx].x;
		data.at<float>(idx,V) = Edge_points_Up[idx].y;
	}
	//Initial Centroids
	Cluster_Center_Up = cv::Mat(MAX_CLUSTER,2,CV_32F);
	//200 800
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
	Cluster_Center_Up.at<float>(0 ,V) =  10 ;
	Cluster_Center_Up.at<float>(1 ,V) =  10 ;
	Cluster_Center_Up.at<float>(2 ,V) =  100;
	Cluster_Center_Up.at<float>(3 ,V) =  100;
	Cluster_Center_Up.at<float>(4 ,V) =  300;
	Cluster_Center_Up.at<float>(5 ,V) =  300;
	Cluster_Center_Up.at<float>(6 ,V) =  400;
	Cluster_Center_Up.at<float>(7 ,V) =  400;
	Cluster_Center_Up.at<float>(8 ,V) =  600;
	Cluster_Center_Up.at<float>(9 ,V) =  600;
	Cluster_Center_Up.at<float>(10,V) =  660;
	Cluster_Center_Up.at<float>(11,V) =  660;
	Cluster_Center_Up.at<float>(12,V) =  720;
	Cluster_Center_Up.at<float>(13,V) =  720;
	Cluster_Center_Up.at<float>(14,V) =  790;
	Cluster_Center_Up.at<float>(15,V) =  790;
	cv::kmeans(data,MAX_CLUSTER, Cluster_bestLabel_Up,
					cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1),
					10,cv::KMEANS_PP_CENTERS,Cluster_Center_Up);
	sortPoints(Cluster_Center_Up);
	for (int i = 0; i < Cluster_Center_Up.rows; i++) {
		float x = Cluster_Center_Up.at<float>(i, 0);
		float y = Cluster_Center_Up.at<float>(i, 1);
		std::cout<< x << ", " << y << std::endl;
	}
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
}

//Corner
void warpPerspectiveTransform(void){
	src_upperPoints.push_back(cv::Point2f(1198, 1043));    //967, 979
	src_upperPoints.push_back(cv::Point2f(2112, 1039));    //3087, 977
	src_upperPoints.push_back(cv::Point2f(1189, 2874));    //967, 1523)
	src_upperPoints.push_back(cv::Point2f(2100, 2924));    //3076, 1523

	dstPoints.push_back(cv::Point2f(0, 0      ));          // top-left
	dstPoints.push_back(cv::Point2f(200, 0    ));          // top-right
	dstPoints.push_back(cv::Point2f(0, 800    ));          // bottom-left
	dstPoints.push_back(cv::Point2f(200, 800  ));          // bottom-right
	perspectiveMatrix = cv::getPerspectiveTransform(src_upperPoints, dstPoints);
	cv::warpPerspective(undistortedsrc_upper, WarpOut, perspectiveMatrix, cv::Size(200, 800));

	std::cout<<perspectiveMatrix<<std::endl;
}

void showImg(){
	cv::namedWindow("Undistort",cv::WINDOW_GUI_NORMAL);
	cv::imshow("Undistort", undistortedsrc_upper);
	cv::namedWindow("Warped", cv::WINDOW_AUTOSIZE);
	cv::imshow("Warped",WarpOut);
	cv::imshow("Corner", Edge);
}

int findOutlierCluster(const cv::Mat& centers) {
	int numClusters = centers.rows;
	double maxDist = 0;
	int outlierIndex = -1;

	for (int i = 0; i < numClusters; i++) {
		double avgDist = 0;
		for (int j = 0; j < numClusters; j++) {
			if (i != j) {
				cv::Point2f pt1 = centers.at<cv::Point2f>(i);
				cv::Point2f pt2 = centers.at<cv::Point2f>(j);
				double dist = cv::norm(pt1 - pt2);
				avgDist += dist;
			}
		}
		avgDist /= (numClusters - 1);
		if (avgDist > maxDist) {
			maxDist = avgDist;
			outlierIndex = i;
		}
	}
	return outlierIndex;
}

/*void createRectangles(const cv::Mat& centers, int outlierIndex) {
	std::vector<cv::Point2f> rectanglePoints;

	// Collect points excluding the outlier
	for (int i = 0; i < centers.rows; i++) {
		if (i != outlierIndex) {
			rectanglePoints.push_back(centers.at<cv::Point2f>(i));
		}
	}
	// Assuming rectanglePoints are ordered properly and grouped by rectangles
	// Typically, you would need to sort or match these points into groups of four
	// that form rectangles, depending on how your initial clustering was done.
	for (size_t i = 0; i < rectanglePoints.size(); i += 4) {
		// Draw or compute the rectangle using the points
		// For simplicity, assume these are top-left, top-right, bottom-right, bottom-left
		cv::rectangle(image, rectanglePoints[i], rectanglePoints[i + 2], cv::Scalar(0, 255, 0), 2);
	}
}*/

void sortPoints(cv::Mat& clusterCenters) {
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