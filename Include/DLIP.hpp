/** @author Jin Kwak/21900031
 * @Created 2024/03/19
 * @Modified -
 * @brief Commonly used functions in Image Processing
 */

#ifndef DLIP_DLIP_HPP
#define DLIP_DLIP_HPP
#include "opencv.hpp"

void showGrayImgHist(cv::Mat src,cv::String str);
void showGrayImgHist(cv::Mat src);
void erode_dilate(cv::Mat src, cv::Mat dst, int N_IDX);
void erode_dilate(cv::Mat src, cv::Mat dst,cv::Size size, int N_IDX);
void dilate_erode(cv::Mat src, cv::Mat dst, int N_IDX);
void dilate_erode(cv::Mat src, cv::Mat dst,cv::Size, int N_IDX);

void image(cv::String str, cv::Mat mat);
#endif //DLIP_DLIP_HPP
