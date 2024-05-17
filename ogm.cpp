#include <opencv.hpp>

int main(){

	cv::Mat src = cv::imread("D:/DLIP/ogm.png");
	cv::namedWindow("OGM-preprocess", cv::WINDOW_AUTOSIZE);
	cv::imshow("OGM-preprocess", src);
	cv::waitKey(0);
	return 0;
}
