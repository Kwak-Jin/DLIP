/** @brief LAB 1 Grayscale Image Segmentation
* @author Jin Kwak/21900031
* @Created 2024/03/22
* @Modified 2024/04/01
*/

#include "DLIP.hpp"
#include <sstream>

#define MAX_BIN_VAL         (int)(255)
#define THRESH_VAL          (double)(180)
#define THRESH_VAL2         (double)(95)
#define KERNEL_SIZE3        (int)(3)
#define KERNEL_SIZE5        (int)(5)
#define KERNEL_SIZE7        (int)(7)
#define GaussianVar         (double)(1.5)
#define BLUE                (cv::Scalar(255,0,0))
#define RED                 (cv::Scalar(0,0,255))
#define GREEN               (cv::Scalar(0,255,0))
#define CYAN                (cv::Scalar(0,255,255))
#define PINK                (cv::Scalar(255,0,255))
#define YELLOW              (cv::Scalar(255,255,0))

/*Define Variables*/
cv::Mat src,src_gray,src_th;
cv::Mat dst_laplace;
cv::Mat dst_masked;
cv::Mat dst_morph;
cv::Mat dst_color;
// Contour Variables
std:: vector<std::vector<cv::Point>>  contours;
std::vector<cv::Vec4i> hierarchy;
cv::Scalar color;
int Component_cnt[5] = {0,};
int False_cnt = 0;
enum Components{
	M6BOLT = 0,
	M5BOLT = 1,
	M6NUT  = 2,
	M5NUT  = 3,
	M5RECT = 4
};

void func_Filter(void);
void func_Threshold(void);
void func_Morphology(void);
void func_Contour(void);

int main(){

	src = cv::imread("Lab_GrayScale_TestImage.jpg",cv::IMREAD_COLOR);
 	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

	func_Filter();

	func_Threshold();

	func_Morphology();

	func_Contour();

	image("Final",src_th);
	cv::waitKey(0);//Pause the program
	return 0;
}

void func_Filter(void){
	cv::medianBlur(src_gray,src_gray, KERNEL_SIZE5);

	cv::GaussianBlur(src_gray,src_gray,
					 cv::Size(KERNEL_SIZE7,KERNEL_SIZE7)
					 ,GaussianVar, GaussianVar);

	cv::Laplacian(src_gray,dst_laplace,CV_16S,
				  KERNEL_SIZE3,1,0,cv::BORDER_DEFAULT);

	dst_laplace. convertTo(dst_laplace,CV_16S);
	src_gray -= (0.7*dst_laplace);
	showGrayImgHist(src_gray);
}

void func_Threshold(void){
    cv::threshold(src_gray,src_th,THRESH_VAL,MAX_BIN_VAL,cv::THRESH_OTSU);
	cv::threshold(src_th,src_th,THRESH_VAL2,MAX_BIN_VAL,cv::THRESH_BINARY);
	//image("Thresh then Laplaced",src_th);
}

void func_Morphology(void){
	cv::dilate(src_th,src_th,cv::Mat::ones(KERNEL_SIZE3,KERNEL_SIZE3,CV_8UC1));
	cv::erode(src_th,src_th,cv::Mat::ones(KERNEL_SIZE5,KERNEL_SIZE5,CV_8UC1));
	cv::morphologyEx(src_th,src_th,cv::MORPH_OPEN,
					 cv::Mat::ones(KERNEL_SIZE3,KERNEL_SIZE3,
								   CV_8UC1),cv::Point(-1,-1),2);
	cv::morphologyEx(src_th,src_th,cv::MORPH_CLOSE,
					 cv::Mat::ones(KERNEL_SIZE7,KERNEL_SIZE7,CV_8UC1),
					 cv::Point(-1,-1),5);
	dst_morph = src_th;
	image("Threshold and morphology",dst_morph);
	//cv::imwrite("Dst_morphology.png",dst_morph);
}

void func_Contour(void){
	std::vector<cv::Scalar> colors(contours.size());
	cv::Mat contour(dst_morph.size(),CV_8U, cv::Scalar(255));
	cv::findContours(dst_morph,contours,
					 hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);
	std::vector<bool> ignore_contour(contours.size(),false);
	cv::drawContours(contour,contours,
					 -1,cv::Scalar (0),cv::FILLED);
	int contourSize = contours.size();          //Number of Contours
	double arc_length =  0;                     //Length of the contour box
	cv::Rect FalseDetect[contourSize];
	/* Make contour rectangles */
	for(int idx = 0; idx<contourSize; idx++)    FalseDetect[idx] = cv::boundingRect(contours[idx]);

	/* Find any rectangles that are inside another rectangles*/
	for(int rectA = 0 ; rectA<contourSize; rectA++){
		for(int rectB = rectA+1 ;rectB<contourSize; rectB++){
			if((FalseDetect[rectA]&FalseDetect[rectB]) == FalseDetect[rectA])     ignore_contour[rectA] = true;
			else if((FalseDetect[rectA]&FalseDetect[rectB])== FalseDetect[rectB]) ignore_contour[rectB] = true;
		}
	}

	/* Print out the number of contours*/
	std::cout<<"The number of detected Industrial components: "<<contourSize<<std::endl;

	/* Segment each rectangle depending on their size */
	for(int idx = 0; idx< contourSize; idx++){
		if(!ignore_contour[idx]){
			arc_length = cv::arcLength(contours[idx], true);
			cv::Rect bounding_rect = FalseDetect[idx];
			if(arc_length >= 500 && arc_length <= 700){
				Component_cnt[M6BOLT] ++;
				color = RED;
			}
			else if(arc_length >= 420 && arc_length < 500){
				Component_cnt[M5BOLT] ++;
				color = BLUE;
			}
			else if(arc_length >= 325 && arc_length < 400){
				False_cnt ++;
				color = GREEN;
			}
			else if(arc_length >= 262 && arc_length < 325){
				Component_cnt[M5NUT] ++;
				color = CYAN;
			}
			else if(arc_length >= 240 && arc_length < 262){
				Component_cnt[M5RECT] ++;
				color = PINK;
			}
			else{
				Component_cnt[M6NUT] ++;
				color = YELLOW;
			}

			cv::rectangle(src, bounding_rect, color, 2, cv::LINE_8, 0);
		}
	}
	//Print the results
	std::cout<<"The number of M6 Bolt: "<<Component_cnt[M6BOLT]<<std::endl;
	std::cout<<"The number of M5 Bolt: "<<Component_cnt[M5BOLT]<<std::endl;
	std::cout<<"The number of M5 HEX NUT: "<<Component_cnt[M5NUT]<<std::endl;
	std::cout<<"The number of M5 RECT NUT: "<<Component_cnt[M5RECT]<<std::endl;
	std::cout<<"The number of M6 HEX NUT: "<<Component_cnt[M6NUT]<<std::endl;
	std::cout<<"The number of False Counts: "<<False_cnt<<std::endl;
	//cv::imwrite("ContouredImage.JPG",src);
	cv::imshow("Contoured",src);
}
