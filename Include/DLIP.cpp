/** @author: Jin Kwak 21900031
 * @created: 2024.03.19
 * @modified: 2024.03.23
 * @about: DLIP functions that can be used in various situations
 */

#include <opencv.hpp>
#include "DLIP.hpp"

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
}

/**
 * @brief Used to smooth the contour, filter out noises (Opening)
 * @param src Source Image
 * @param dst Output image
 * @param N_IDX Number of iteration
 */
void erode_dilate(cv::Mat src, cv::Mat dst, int N_IDX){
	for(int i = 0; i<N_IDX;i++){
		cv::erode(src,dst,cv::Mat());
		cv::dilate(src,dst,cv::Mat());
	}
}

/**
 * @brief Used to smooth the contour, filter out noises (Opening)
 * @param src Source Image
 * @param dst Output image
 * @param size Matrix size
 * @param N_IDX Number of iteration
 */
void erode_dilate(cv::Mat src, cv::Mat dst,cv::Size size, int N_IDX){
	for(int i = 0; i<N_IDX;i++){
		cv::erode(src,dst,cv::Mat::zeros(size,CV_8UC1));
		cv::dilate(src,dst,cv::Mat::zeros(size, CV_8UC1));
	}
}

/**
 * @brief Used to smooth the holes (Closing)
 * @param src Source Image
 * @param dst Output image
 * @param N_IDX Number of iteration
 */
void dilate_erode(cv::Mat src, cv::Mat dst, int N_IDX){
	for(int i = 0; i<N_IDX;i++){
		cv::dilate(src,dst,cv::Mat());
		cv::erode(src,dst,cv::Mat());
	}
}

/**
 * @brief Used to smooth the holes (Closing)
 * @param src Source Image
 * @param dst Output image
 * @param size Matrix size
 * @param N_IDX Number of iteration
 */
void dilate_erode(cv::Mat src, cv::Mat dst,cv::Size size, int N_IDX){
	for(int i = 0; i<N_IDX;i++){
		cv::dilate(src,dst,cv::Mat::zeros(size, CV_8UC1));
		cv::erode(src,dst,cv::Mat::zeros(size,CV_8UC1));
	}
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
