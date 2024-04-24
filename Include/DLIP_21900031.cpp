/** @author: Jin Kwak 21900031
 * @created: 2024.03.19
 * @modified: 2024.04.02
 * @about: DLIP functions that can be used in various situations
 */

#include "DLIP_21900031.hpp"

/**
 * @param src: Should be grayscale
 * @param str: Name of the image e.g. Histogram.JPG
 * @return Shows histogram and grayscale image source and Saves the histogram
 */
void showGrayImgHist(cv::Mat src,cv::String str){
	/*********************************Histogram of the Original Image Source*******************************************/
	const int* channel_numbers = { 0 };
	cv::MatND hist;
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 256;

	calcHist(&src, 1, channel_numbers, cv::Mat(), hist, 1, &number_bins, &channel_ranges);

	// plot the histogram
	int hist_w = src.cols;
	int hist_h = src.rows;
	int bin_w = cvRound((double)hist_w / number_bins);

	cv::Mat hist_img(hist_h, hist_w, CV_8UC1, cv::Scalar::all(255));
	normalize(hist, hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for (int i = 1; i < number_bins; i++){
		cv::line(hist_img, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
		         cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
		         cv::Scalar(0, 0, 0), 1, 8, 0);
	}
	cv::namedWindow("Original", cv::WINDOW_NORMAL);
	cv::imshow("Original", src);
	cv::namedWindow("Histogram", cv::WINDOW_NORMAL);
	cv::imshow("Histogram", hist_img);
	cv::imwrite(str,hist_img);
}

void showGrayImgHist(cv::Mat src){
	/*********************************Histogram of the Original Image Source*******************************************/
	const int* channel_numbers = { 0 };
	cv::MatND hist;
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 256;

	calcHist(&src, 1, channel_numbers, cv::Mat(), hist, 1, &number_bins, &channel_ranges);

	// plot the histogram
	int hist_w = src.cols;
	int hist_h = src.rows;
	int bin_w = cvRound((double)hist_w / number_bins);

	cv::Mat hist_img(hist_h, hist_w, CV_8UC1, cv::Scalar::all(255));
	cv::normalize(hist, hist, 0, hist_img.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for (int i = 1; i < number_bins; i++){
		cv::line(hist_img, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
		         cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
		         cv::Scalar(0, 0, 0), 1, 8, 0);
	}
	cv::namedWindow("Original", cv::WINDOW_NORMAL);
	cv::imshow("Original", src);
	cv::namedWindow("Histogram", cv::WINDOW_NORMAL);
	cv::imshow("Histogram", hist_img);
	cv::imwrite("FilterHistogram.JPG",hist_img);
}

/**
 * @brief Image loading with Auto named window
 * @param str Image Name
 * @param mat Image
 */
void image(cv::String str, cv::Mat mat){
	cv::namedWindow(str, cv::WINDOW_AUTOSIZE);
	cv::imshow(str,mat);
}


// /**
//  *
//  * @param img source to set roi
//  * @param xratio x direction cut ratio
//  * @param yratio y direction cut ratio
//  * @param x start pixel x
//  * @param y start pixel y
//  */
// void setROI(cv::Mat img, double xratio, double yratio, int x, int y){
// 	cv::Rect regionofInterest(x,y, ((int)(img.rows*xratio)),((int)(img.cols*yratio)));
// 	if(x+ ((int)(img.rows*xratio))>= img.rows || y+ ((int)(img.cols*yratio))) img = img;
// 	else img = img(regionofInterest);
// }

/**
 * @param _img BGR Image source
 * @param _threshVal1 Lower Threshold Value
 * @param _threshVal2 Upper Threshold Value(Usually 255)
 * @brief Contours(Yellow) on Original Source Image
 */
void CVcontour(cv::Mat _img,int _threshVal1,int _threshVal2){
	cv::Mat gray, thresh;
	cv::cvtColor(_img, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, thresh, _threshVal1, _threshVal2, cv::THRESH_BINARY);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	drawContours(_img, contours, -1, cv::Scalar(0, 255, 255), 2);

}
