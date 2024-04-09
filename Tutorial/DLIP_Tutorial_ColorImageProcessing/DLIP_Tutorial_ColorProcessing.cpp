/**
 * @about Color Image Processing
 * @created 2024.04.09
 * @author Jin Kwak/21900031
 */

// H 0~179
// S 0~250
// V 0~250

#include <opencv.hpp>
#include <iostream>
cv::Mat image;
cv::Point origin;
cv::Rect selection;
bool selectObject = false;
bool trackObject = false;
int hmin = 1, hmax = 179, smin = 30, smax = 255, vmin = 0, vmax = 255;

static void onMouse(int event, int x, int y, int, void*);

int main(){
	cv::Mat image_disp, hsv, hue, mask, dst;
	std::vector<std::vector<cv::Point> > contours;

	image = cv::imread("color_ball.jpg");
	image.copyTo(image_disp);

	cv::Mat dst_track = cv::Mat::zeros(image.size(), CV_8UC3);

	// TrackBar 설정
	cv::namedWindow("Source", 0);
	cv::setMouseCallback("Source", onMouse, 0);
	cv::createTrackbar("Hmin", "Source", &hmin, 179, 0);
	cv::createTrackbar("Hmax", "Source", &hmax, 179, 0);
	cv::createTrackbar("Smin", "Source", &smin, 255, 0);
	cv::createTrackbar("Smax", "Source", &smax, 255, 0);
	cv::createTrackbar("Vmin", "Source", &vmin, 255, 0);
	cv::createTrackbar("Vmax", "Source", &vmax, 255, 0);

	cv::VideoCapture cap(0);
	if (!cap.isOpened()){	// if not success, exit the programm
		std::cout << "Cannot open the video cam\n";
		return -1;
	}

	while (true){
		bool bSuccess = cap.read(image);
		if (!bSuccess){
			std::cout << "Cannot find a frame from  video stream\n";
			break;
		}
		/******** Convert BGR to HSV ********/
		cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
		cv::imshow("Source",image);
		/******** Add Pre-Processing such as filtering etc  ********/


		/// set dst as the output of InRange
		inRange(hsv, cv::Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
		        cv::Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst);


		/******** Add Post-Processing such as morphology etc  ********/
		// YOUR CODE GOES HERE
		// YOUR CODE GOES HERE

		cv::namedWindow("InRange", 0);
		imshow("InRange", dst);

		/// once mouse has selected an area bigger than 0
		if (trackObject){
			trackObject = false;					// Terminate the next Analysis loop
			cv::Mat roi_HSV(hsv, selection); 			// Set ROI by the selection box
			cv::Scalar means, stddev;
			meanStdDev(roi_HSV, means, stddev);
			std::cout << "\n Selected ROI Means= " << means << " \n stddev= " << stddev;

			// Change the value in the trackbar according to Mean and STD //
			hmin = MAX((means[0] - stddev[0]), 0);
			hmax = MIN((means[0] + stddev[0]), 179);

			vmin = MAX((means[2] - stddev[2]), 0);
			vmax = MIN((means[2] + stddev[2]), 250);
			smin = MAX((means[1] - stddev[1]), 0);
			smax = MIN((means[1] + stddev[1]), 250);
			cv::setTrackbarPos("Hmin", "Source", hmin);
			cv::setTrackbarPos("Hmax", "Source", hmax);
			cv::setTrackbarPos("Vmax", "Source", vmax);
			cv::setTrackbarPos("Vmin", "Source", vmin);
			cv::setTrackbarPos("Smax", "Source", smax);
			cv::setTrackbarPos("Smin", "Source", smin);
		}
		if (selectObject && selection.area() > 0){  // Left Mouse is being clicked and dragged
			// Mouse Drag을 화면에 보여주기 위함
			cv::Mat roi_RGB(image, selection);
			bitwise_not(roi_RGB, roi_RGB);
		}
		imshow("Source", image);
		///  Find All Contour   ///
		findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		if (contours.size()>0){
			/// Find the Contour with the largest area ///
			double maxArea = 0;
			int maxArea_idx = 0;

			for (int i = 0; i < contours.size(); i++)
				if (contourArea(contours[i]) > maxArea){
					maxArea = contourArea(contours[i]);
					maxArea_idx = i;
				}

			///  Draw the max Contour on Black-background  Image ///
			cv::Mat dst_out =cv::Mat::zeros(dst.size(), CV_8UC3);
			drawContours(dst_out, contours, maxArea_idx, cv::Scalar(0, 0, 255), 2, 8);

			cv::namedWindow("Contour", 0);
			cv::imshow("Contour", dst_out);

			/// Draw the Contour Box on Original Image ///
			cv::Rect boxPoint = boundingRect(contours[maxArea_idx]);
			cv::rectangle(image, boxPoint, cv::Scalar(255, 0, 255), 3);
			cv::namedWindow("Contour_Box", 0);
			cv::imshow("Contour_Box", image);

			/// Continue Drawing the Contour Box  ///
			///  	Fade out 	 ///
			cv::rectangle(dst_track, boxPoint, cv::Scalar(255, 0, 255), 3);
			dst_track *= 0.85;
			cv::namedWindow("Contour_Track", 0);
			imshow("Contour_Track", dst_track);
		}

		char c = (char)cv::waitKey(10);
		if (c == 27) break;
	}
	return 0;
}

/// On mouse event
static void onMouse(int event, int x, int y, int, void*){
	if (selectObject){  // for any mouse motion
		selection . x = MIN(x, origin . x);
		selection . y = MIN(y, origin . y);
		selection . width = abs(x - origin . x) + 1;
		selection . height = abs(y - origin . y) + 1;
		selection &= cv::Rect(0, 0, image . cols,
		                      image . rows);  /// Bitwise AND  check selectin is within the image coordinate
	}
	switch (event){
		case cv::EVENT_LBUTTONDOWN:
			selectObject = true;
			origin = cv::Point(x, y);
			break;
		case cv::EVENT_LBUTTONUP:
			selectObject = false;
			if (selection.area())
				trackObject = true;
			break;
	}
}