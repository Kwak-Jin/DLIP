#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", WINDOW_AUTOSIZE);
	Mat frame;
	Mat flippedFrame;
	int isFlip = 1;
	while (1){

		/*  read a new frame from video  */
		bool bSuccess = cap.read(frame);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}
		flip(frame,flippedFrame,isFlip);
		imshow("MyVideo", flippedFrame);
		if(waitKey(30)=='h'){
			isFlip *= -1;
		}
		if (waitKey(30) == 27)
		{
			cout << "ESC key is pressed by user\n";
			break;
		}
	}
}
