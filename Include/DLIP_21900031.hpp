/** @author Jin Kwak/21900031
 * @Created 2024/03/19
 * @Modified -
 * @brief Commonly used functions in Image Processing
 */

#ifndef DLIP_DLIP_HPP
#define DLIP_DLIP_HPP
#include "opencv.hpp"
#include <iostream>
#include <cmath>
void showGrayImgHist(cv::Mat src,cv::String str);
void showGrayImgHist(cv::Mat src);
void image(cv::String str, cv::Mat mat);
// void setROI(cv::Mat img, double xratio, double yratio, int x, int y);
void CVcontour(cv::Mat _img,int _threshVal1,int _threshVal2);
#endif //DLIP_DLIP_HPP
