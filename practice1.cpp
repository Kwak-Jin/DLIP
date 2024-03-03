/**
* @brief Cmake and OpenCV practice
* @author Jin Kwak/21900031
* @created 24/03/01
* @modified NONE
*/

#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
	cv::Mat source;
	source = cv::imread("C:/Users/jinkwak/Downloads/testImage.JPG",1);
	if(source.empty()){
		std::cout<<"Empty"<<std::endl;
	}
	namedWindow("DemoWIndow", cv::WINDOW_AUTOSIZE); //WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window

	if (!source.empty())imshow("DemoWIndow", source); // Show image

	cv::waitKey(0);//Pause the program
	return 0;
}

