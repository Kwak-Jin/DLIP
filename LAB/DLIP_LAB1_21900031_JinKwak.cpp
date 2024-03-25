/** @brief LAB 1 Grayscale Image Segmentation
* @author Jin Kwak/21900031
* @Created 2024/03/22
* @Modified -
*/

#include "DLIP.hpp"
#define MAX_BIN_VAL (int)(255)
#define THRESH_VAL  (double)(190)
#define KERNEL_SIZE (int)(5)
#define THRESH_VAL2 (double)(95)
/* Steps:
 * --> Gaussian
 * --> Thresholding
 * --> Opening
 * --> Closing
 * */
int main(){
	cv::Mat src = cv::imread("Lab_GrayScale_TestImage.jpg",cv::IMREAD_COLOR);
	cv::Mat src_gray,source_equalized,src_th;
	cv::Mat dst_masked   ;

	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	//cv::equalizeHist(src_gray,src_gray);


	//cv::Laplacian(src_gray, dst_laplacian, CV_16S, KERNEL_SIZE, 1, 0, cv::BORDER_DEFAULT);
	//src_gray -= (1*dst_laplacian);
	cv::threshold(src_gray,src_th,THRESH_VAL,MAX_BIN_VAL,cv::THRESH_OTSU);
	src_th -= (1*src_gray);
	image("Thresh Laplaced",src_th);

	cv::threshold(src_th,src_th,THRESH_VAL2,MAX_BIN_VAL,cv::THRESH_BINARY);
	image("Med",src_th);
	//erode_dilate(src_th,src_th,3);

	for(int idx = 0;idx<1;idx++)	cv::dilate(src_th,src_th,cv::Mat::ones(cv::Size(3,3),CV_16S));
	dilate_erode(src_th,src_th,100);
	image("Final",src_th);
	cv::waitKey(0);//Pause the program
	return 0;
}
