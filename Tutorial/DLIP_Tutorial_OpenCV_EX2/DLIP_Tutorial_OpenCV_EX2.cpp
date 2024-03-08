/**
 *
 */

#include <iostream>
#include <opencv.hpp>
int main(){
	cv::String logo_dir = "D:\\DLIP\\Image\\HGU_logo.jpg";
	cv::Mat HGU_logo = cv::imread(logo_dir,1);
	cv::Mat gray_HGU ;
	cv::cvtColor(HGU_logo,gray_HGU,cv::COLOR_BGR2GRAY);
	int N_rows = HGU_logo.rows;
	int N_cols = HGU_logo.cols;
	double intensity = 0;

	int N_size = N_rows*N_cols;
	for(int rows= 0; rows<N_rows;rows++){
		for(int cols = 0; cols < N_cols; cols ++){
			 intensity+= (double)(HGU_logo.at<uchar>(rows,cols));
		}
	}
	intensity = (double)(intensity/N_size);
	std::cout<<"intensity = "<<intensity<<std::endl;
}