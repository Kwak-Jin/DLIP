/** @brief DLIP Hough Circle Detection
* @author Jin Kwak 21900031
* @created 2024.04.16
*/

#include <DLIP.hpp>

using namespace cv;
int main(int argc, char** argv)
{
	Mat src, gray;

	String filename = "D:\\DLIP\\Image\\EdgeLineImages\\eyepupil.png";

	/* Read the image */
	src = imread(filename, 1);

	if (!src.data)
	{
		printf(" Error opening image\n");
		return -1;
	}

	cvtColor(src, gray, COLOR_BGR2GRAY);
	equalizeHist(gray,gray);
	/* smooth it, otherwise a lot of false circles may be detected */
	GaussianBlur(gray, gray, Size(7, 7), 3, 3);

	std::vector<Vec3f> circles;

	HoughCircles(gray, circles, 3, 2, gray.rows / 4, 140, 70);

	for (size_t i = 0; i < circles.size(); i++){
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		/* draw the circle center */
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		/* draw the circle outline */
		circle(src, center, radius, Scalar(255, 0, 255), 3, 8, 0);
	}

	namedWindow("circles", 1);
	imshow("circles", src);

	/* Wait and Exit */
	waitKey();
	return 0;
}