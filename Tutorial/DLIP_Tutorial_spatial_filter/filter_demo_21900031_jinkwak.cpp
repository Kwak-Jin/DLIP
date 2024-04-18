/* ------------------------------------------------------ /
*Image Proccessing with Deep Learning
* OpenCV : Filter Demo
* Created : 2021 - Spring
* Author: Jin Kwak/21900031
------------------------------------------------------ */

#include <opencv.hpp>
#include <iostream>

using namespace std;

int main(){
	cv::Mat src, dst_blur, dst_gaussian,dst_median,dst_laplacian,dst_2Dconv;
	src = cv::imread("D:\\DLIP\\Image\\KakaoTalk_20240311_131533056.jpg", 0);

	int i = 5;
	cv::Size kernelSize = cv::Size(i, i);

	//Original
	cv::imshow("Original", src);
	/* Blur */
	cv::blur(src, dst_blur, cv::Size(i, i), cv::Point(-1, -1));
	cv::namedWindow("Blur", cv::WINDOW_NORMAL);
	cv::imshow("Blur", dst_blur);


	/* Gaussian Filter */
	cv::GaussianBlur(src, dst_gaussian, cv::Size(i,i), 0,0);
	namedWindow("Gaussian", cv::WINDOW_NORMAL);
	imshow("Gaussian", dst_gaussian);

	/* Median Filter */
	cv::medianBlur(src,dst_median,i);
	namedWindow("Median", cv::WINDOW_AUTOSIZE);
	imshow("Median", dst_median);

	/* Laplacian Filter */
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	cv::Laplacian(src, dst_laplacian, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
	src.convertTo(src, CV_16S);
	cv::Mat result_laplcaian = src - dst_laplacian;
	result_laplcaian.convertTo(result_laplcaian, CV_8U);
	namedWindow("Laplacian", cv::WINDOW_AUTOSIZE);
	cv::imshow("Laplacian", result_laplcaian);

	/* 2D Convolution of a filter kernel */
	/* Design a normalized box filter kernel 5 by 5 */
	src.convertTo(src, CV_8UC1);

	cv::Mat kernel;
	delta = 0;
	ddepth = -1;
	kernel_size = 5;
	cv::Point anchor = cv::Point(-1, -1);
	kernel = cv::Mat::zeros(kernel_size,kernel_size,CV_32F); //(kernel_size*kernel_size);
	for(int i = 0; i<kernel_size; i++) {
		kernel.at<float>(0,i)= -1;
		kernel.at<float>(kernel_size-1,i)= 1;
	}
	cv::filter2D(src,dst_2Dconv,ddepth,kernel);
	namedWindow("Conv2D", cv::WINDOW_AUTOSIZE);
	cv::imshow("Conv2D", dst_2Dconv);


	cv::waitKey(0);
	return 0;
}
