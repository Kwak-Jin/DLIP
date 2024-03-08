//
// Created by jinkwak on 2024-03-08.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// Load the image
	Mat src = imread("D:\\DLIP\\Image\\HGU_logo.jpg", 0);

	if (src.empty())
	{
		cout << "Error: Couldn't open the image." << endl;
		return -1;
	}

	// Initialize dst with the same size as srcGray
	Mat dst = Mat::ones(src.size(), src.type());
	dst = 255*dst;
	dst -= src;
	// Invert the colors by accessing each pixel

	// Display the original and inverted images
	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);

	return 0;
}
