#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;
#define DELAY 100

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
			if (isFlip == 1) isFlip = 0;
			else if(isFlip == 0) isFlip = 1;
		}
		else if (waitKey(30) == 27){
			cout << "ESC key is pressed by user\n";
			break;
		}
	}
}
