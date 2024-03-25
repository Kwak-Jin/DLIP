/**
* @brief Cmake and OpenCV practice
* @author Jin Kwak/21900031
* @created 24/03/01
* @modified NONE
*/

#include <iostream>
#include "opencv.hpp"

int main(){
	//Bookmark
	cv::Mat	source1 = cv::imread("C:/Users/jinkwak/Downloads/testImage.JPG",1);
	cv::Mat source2 = cv::imread("C:\\Users\\jinkwak\\Pictures\\Screenshots\\2024-03-01 180035.png");
	if(source1.empty()||source2.empty()) std::cout<<"Empty"<<std::endl;
	else{
		namedWindow("DemoWindow", cv::WINDOW_AUTOSIZE); //WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window
		imshow("DemoWindow", source1); // Show image
		imshow("DemoWindow2",source2);
	}

	cv::waitKey(0);//Pause the program
	return 0;
}

