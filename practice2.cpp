/**@brief OpenCV practice
 * @author Jin Kwak/21900031
 * @created 24/03/03
 * @modified NONE
 */

#include <iostream>
#include "opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	VideoCapture cap(0); // open the video camera no. 0

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}
	namedWindow("MyVideo",WINDOW_AUTOSIZE); //create a window called "MyVideo"
	Mat grey;
	Mat Cann;
	while (1){
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video
		if (!bSuccess){ /*if not success, break loop*/
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		cvtColor(frame,grey,COLOR_BGR2GRAY);
		Canny(grey,Cann,30,120,3, false);
		imshow("MyVideo", Cann); //show the frame in "MyVideo" window
		//imshow("MyVideo",Cann);
		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;
}