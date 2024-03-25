/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Filter Demo - Video
* Created: 2021-Spring
------------------------------------------------------*/

#include <opencv.hpp>
#include <iostream>
#include <conio.h>
#define ESC 27
enum FILTER_TYPE{
	Default = 0,
	Blur = 1,
	Gaussian = 2,
	Median = 3,
	Laplacian = 4,
	Bilateral = 5
};

using namespace std;

int main(){
	/*  open the video camera no.0  */
	cv::VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", cv::WINDOW_AUTOSIZE);

	int key = 0;
	int kernel_size = 21;
	int filter_type = 0;
	cv::Mat src, dst;
	while (1){
		/*  read a new frame from video  */
		bool bSuccess = cap.read(src);
		if (!bSuccess){
			cout << "Cannot find a frame from  video stream\n";
			break;
		}
		if(key = cv::waitKey(30)){
			if (key == ESC){ // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
				cout << "ESC key is pressed by user\n";
				break;
			}
			else if(key == 'b'|| key == 'B')	filter_type = Blur     ;
			else if(key == 'g'|| key =='G' )    filter_type = Gaussian ;
			else if(key == 'm'|| key =='M' )    filter_type = Median   ;
			else if(key == 'L'|| key == 'l')    filter_type = Laplacian;
			else if(key == 'D'|| key =='d' )    filter_type = Default  ;
			else if(key == '1'|| key =='!' )    filter_type = Bilateral;
			else if(key == 'U'|| key =='u' )      kernel_size+=2       ;
			else if(key == ' '&&kernel_size >3)   kernel_size-=2       ;
		}
		else filter_type= filter_type;

		switch(filter_type){
			case Blur:
				cv::blur(src, dst, cv::Size(kernel_size, kernel_size));
				cout<< "Blur"<<endl;
				break;
			case Gaussian:
				cv::GaussianBlur(src, dst, cv::Size(kernel_size,kernel_size), 0,0);
				cout<< "Gaussian"<<endl;
				break;
			case Median:
				cv::medianBlur(src,dst,kernel_size);
				cout<< "Median"<<endl;
				break;
			case Laplacian:
				cv::Laplacian(src, dst, CV_16S, kernel_size, 1, 0, cv::BORDER_DEFAULT);
				src.convertTo(src, CV_16S);
				dst = src- dst;
				dst. convertTo(dst,CV_8U);
				cout<< "Laplacian"<<endl;
				break;
			//case Bilateral:
			//	cv::bilateralFilter(src,dst, kernel_size, 3,kernel_size);
			//	cout<< "Bilateral"<<endl;
			//	break;
			default:
				dst = src;
				cout<< "Default"<<endl;
				break;
		}
		imshow("MyVideo", dst);
	}
	return 0;
}