/** @author Jin Kwak/21900031
*  @brief OpenCV Image
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

	cv::Mat dist;
	cv::String fileName = "writeImage.jpg";
	cv::flip(src, dist, 0);
	imwrite(fileName, dist);

	/*  display image  */
	namedWindow("src", cv::WINDOW_AUTOSIZE);
	imshow("src", src);

	namedWindow("Flipped Image", cv::WINDOW_AUTOSIZE);
	imshow("Flipped Image", dist);

	cv::waitKey(0);
}
