#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv.hpp>
#include <stdlib.h>
#include <stdio.h>


/// Global variables

cv::Mat src, src_gray;
cv::Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	cv::blur(src_gray, detected_edges, cv::Size(3, 3));


	/// Canny detector
	cv::Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = cv::Scalar::all(0);

	src.copyTo(dst, detected_edges);
	cv::imshow(window_name, dst);
}


/** @function main */
int main(int argc, char** argv)
{
	/// Load an image
	const char* filename = "../../../Image/EdgeLineImages/coins.png";
	//const char* filename = "../images/TrafficSign1.png";

	/// Read the image
	src = cv::imread(filename, 1);

	if (!src.data)
	{
		return -1;
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale
	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

	/// Create a window
	namedWindow(window_name, cv::WINDOW_AUTOSIZE);

	/// Create a Trackbar for user to enter threshold
	cv::createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	/// Show the image
	CannyThreshold(0, 0);

	/// Wait until user exit program by pressing a key
	cv::waitKey(0);

	return 0;
}