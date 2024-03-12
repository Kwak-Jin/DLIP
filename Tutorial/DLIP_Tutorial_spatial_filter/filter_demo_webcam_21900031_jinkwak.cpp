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
	Laplacian = 4
};

using namespace std;

int main()
{
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
			else if (key == 'b' || key == 'B')	filter_type = Blur;
			else if(key == 'g'|| key =='G')     filter_type = Gaussian;
			else if(key =='m'|| key =='M')      filter_type = Median;
			else if(key == 'L'||key == 'l')     filter_type = Laplacian;
			else if(key == 'D'||key =='d')      filter_type = Default;
			else if(key == 'U'||key =='u') {
				if(cv::waitKey(30)== 'P'||cv::waitKey(30)== 'p') kernel_size+=2;
			}
			else if(key == ' ' && kernel_size >3)  kernel_size-=2;
			else                                filter_type = filter_type;
		}
		else filter_type= filter_type;

		switch(filter_type){
			case Blur:
				cv::blur(src, dst, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
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
				cv::Laplacian(src, dst, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
				cout<< "Laplacian"<<endl;
				break;
			default:
				dst = src;
				cout<< "Default"<<endl;
				break;
		}
		imshow("MyVideo", dst);
	}
	return 0;
}

//#include <iostream>
//#include <opencv.hpp>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	/*  read image  */
//	Mat img = imread("D:\\DLIP\\Image\\HGU_logo.jpg");
//	imshow("img", img);
//
//	/*  Crop(Region of Interest)  */
//	Rect r(10, 10, 150, 150);	 // (x, y, width, height)
//	Mat roiImg = img(r);
//	imshow("roiImg", roiImg);
//
//	/*  Rotate  */
//	Mat rotImg;
//	rotate(img, rotImg, ROTATE_90_CLOCKWISE);
//	imshow("rotImg", rotImg);
//
//	/*  Resize  */
//	Mat resizedImg;
//	resize(img, resizedImg, Size(1000, 100));
//	imshow("resizedImg", resizedImg);
//
//	waitKey(0);
//}