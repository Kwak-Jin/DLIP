/** @author Jin Kwak/21900031
*  @brief OpenCV Image Processing Flip
*  @created 2024/03/08
*/

#include <iostream>
#include <opencv.hpp>

using namespace std;

int main()
{
	/*  read image  */
	cv::String HGU_logo = "D:\\DLIP\\Image\\HGU_logo.jpg";
	cv::Mat src = cv::imread(HGU_logo);
	cv::Mat src_gray = cv::imread(HGU_logo, 0);  // read in grayscale
	cv::cvtColor(src, src_gray,cv::COLOR_BGR2GRAY);
	/*  write image  */
	cv::String fileName = "writeImage.jpg";
	imwrite(fileName, src);

	/*  display image  */
	namedWindow("src", cv::WINDOW_AUTOSIZE);
	imshow("src", src);

	namedWindow("src_gray", cv::WINDOW_AUTOSIZE);
	imshow("src_gray", src_gray);

	cv::waitKey(0);
}