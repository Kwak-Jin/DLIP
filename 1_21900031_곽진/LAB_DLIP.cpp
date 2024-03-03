/** @brief This is a practice source for Cmake
*  @author Jin Kwak/21900031
*  @created 24/03/03
*  @modified NONE
*/

#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
	cv::VideoCapture cap(0); // open the video camera no. 0

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}
	namedWindow("MyVideo",cv::WINDOW_AUTOSIZE); //create a window called "MyVideo"

	while (1)
	{
		cv::Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video
		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		imshow("MyVideo", frame); //show the frame in "MyVideo" window


		if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}