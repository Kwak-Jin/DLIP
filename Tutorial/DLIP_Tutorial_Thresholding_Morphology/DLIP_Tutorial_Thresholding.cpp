/**
* @author Jin Kwak/21900031
* @created 2024/03/19
*/


#include <iostream>
#include "DLIP.hpp"

using namespace std;

int main(){

	cv::Mat src, src_gray, dst, dst_morph;

	src = cv::imread("D:\\DLIP\\Image\\testImage\\coin.jpg", 0);    // Load an image

	if (src.empty()){					// Load image check
		cout << "File Read Failed : src is empty" << endl;
		cv::waitKey(0);
	}
	// Create a window to display results
	cv::namedWindow("DemoWindow", cv::WINDOW_AUTOSIZE); //CV_WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window
	if (!src.empty())imshow("DemoWindow", src); // Show image
	/* threshold_type
	0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted  */
	int threshold_value = 100       ;
	int const max_value = 255       ;
	int const max_type  = 4         ;
	int const max_BINARY_value = 255;

	threshold(src, dst, threshold_value, max_BINARY_value, cv::THRESH_TOZERO);

	// Create a window to display results
	namedWindow("ThreshWindow", cv::WINDOW_AUTOSIZE); //CV_WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window
	cv::imshow("ThreshWindow", dst);                   // Show image
	//GaussianFilterImshow(src, dst);
	cv::waitKey(0);             //Pause the program
	return 0;
}
