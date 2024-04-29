// #include <iostream>
// #include <opencv.hpp>
// #include <string>
//
// cv::Mat src;
// cv::Mat src_gray;
// cv::Mat src_white;
// cv::Mat src_black;
//
// // Contour Variables
// std:: vector<std::vector<cv::Point>>  contours;
// std::vector<cv::Vec4i> hierarchy;
//
// int main() {
// 	// Already Calibrated
// 	src = cv::imread("D:/DLIP/cmake-build-debug/Image/LAB2/LAB2Image1.jpg");
// 	cv::cvtColor(src,src_gray,cv::COLOR_BGR2GRAY);
// 	for(int idx = 0; idx<50; idx++) cv::medianBlur(src_gray,src_gray,3);
// 	for(int idx = 0; idx<10; idx++) cv::medianBlur(src_gray,src_gray,5);
//
// 	cv::threshold(src_gray,src_white,200,255,cv::THRESH_BINARY);
// 	cv::threshold(src_gray,src_black,40,255,cv::THRESH_BINARY_INV);
// 	cv::bitwise_or(src_white,src_black,src_black);
//
// 	for(int idx= 0; idx<10; idx++){
// 		cv::erode(src_black,src_black,5);
// 		cv::dilate(src_black,src_black,5);
// 	}
// 	cv::findContours(src_black,contours,
// 					 hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);
// 	cv::drawContours(src,contours,-1,cv::Scalar(255,0,255),cv::FILLED);
// 	int maxAreaIdx = 0;
// 	if(contours.size() <=0) {
// 		std::cout<<"No contour detected"<<std::endl;
// 		return 0;
// 	}
// 	cv::namedWindow("src",cv::WINDOW_GUI_NORMAL);
// 	cv::imshow("src",src);
// 	cv::namedWindow("src1",cv::WINDOW_GUI_NORMAL);
// 	cv::imshow("src1",src_black);
// 	cv::waitKey(0);
// 	return 0;
// }

#include "opencv.hpp"

#include <iostream>

using namespace std;
using namespace cv;

static void help(char** argv)
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo program shows how perspective transformation applied on an image, \n"
        << "Using OpenCV version " << CV_VERSION << endl;

    cout << "\nUsage:\n" << argv[0] << " [image_name -- Default sample6.jpg]\n" << endl;

    cout << "\nHot keys: \n"
        "\tESC, q - quit the program\n"
        "\tr - change order of points to rotate transformation\n"
        "\tc - delete selected points\n"
        "\ti - change order of points to inverse transformation \n"
        "\nUse your mouse to select a point and move it to see transformation changes" << endl;
}

static void onMouse(int event, int x, int y, int, void*);

String windowTitle = "Perspective Transformation Demo";
String labels[4] = { "TL","TR","BR","BL" };
vector<Point2f> roi_corners;
vector<Point2f> midpoints(4);
vector<Point2f> dst_corners(4);
int roiIndex = 0;
bool dragging;
int selected_corner_index = 0;
bool validation_needed = true;

int main(int argc, char** argv)
{
    help(argv);
    CommandLineParser parser(argc, argv, "{@input|D:/DLIP/cmake-build-debug/Image/LAB2/WithCoinUp.jpg|}");

    string filename = parser.get<string>("@input");
    Mat original_image = imread(filename);
    Mat image;

    float original_image_cols = (float)original_image.cols;
    float original_image_rows = (float)original_image.rows;
    roi_corners.push_back(Point2f((float)(original_image_cols / 1.70), (float)(original_image_rows / 4.20)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.15), (float)(original_image.rows / 3.32)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.33), (float)(original_image.rows / 1.10)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.93), (float)(original_image.rows / 1.36)));

    namedWindow(windowTitle, WINDOW_GUI_NORMAL);
    namedWindow("Warped Image", WINDOW_GUI_NORMAL);
    moveWindow("Warped Image", 20, 20);
    moveWindow(windowTitle, 330, 20);

    setMouseCallback(windowTitle, onMouse, 0);
    Mat warped_image;

    bool endProgram = false;
    while (!endProgram){
        if (validation_needed && roi_corners.size() < 4){
            validation_needed = false;
            image = original_image.clone();

            for (size_t i = 0; i < roi_corners.size(); ++i){
                circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);

                if (i > 0)
                {
                    line(image, roi_corners[i - 1], roi_corners[(i)], Scalar(0, 0, 255), 2);
                    circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
                    putText(image, labels[i].c_str(), roi_corners[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
                }
            }
            imshow(windowTitle, image);
        }

        if (validation_needed && roi_corners.size() == 4){
            image = original_image.clone();
            for (int i = 0; i < 4; ++i){
                line(image, roi_corners[i], roi_corners[(i + 1) % 4], Scalar(0, 0, 255), 2);
                circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
                putText(image, labels[i].c_str(), roi_corners[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
            }

            imshow(windowTitle, image);

            midpoints[0] = (roi_corners[0] + roi_corners[1]) / 2;
            midpoints[1] = (roi_corners[1] + roi_corners[2]) / 2;
            midpoints[2] = (roi_corners[2] + roi_corners[3]) / 2;
            midpoints[3] = (roi_corners[3] + roi_corners[0]) / 2;

            dst_corners[0].x = 0;
            dst_corners[0].y = 0;
            dst_corners[1].x = (float)norm(midpoints[1] - midpoints[3]);
            dst_corners[1].y = 0;
            dst_corners[2].x = dst_corners[1].x;
            dst_corners[2].y = (float)norm(midpoints[0] - midpoints[2]);
            dst_corners[3].x = 0;
            dst_corners[3].y = dst_corners[2].y;

            Size warped_image_size = Size(cvRound(dst_corners[2].x), cvRound(dst_corners[2].y));

            Mat M = getPerspectiveTransform(roi_corners, dst_corners);

            warpPerspective(original_image, warped_image, M, warped_image_size); // do perspective transformation

            imshow("Warped Image", warped_image);
        }

        char c = (char)waitKey(10);

        if ((c == 'B') || (c == 'b') || (c == 27))  endProgram = true;
        if ((c == 'c') || (c == 'C'))  roi_corners.clear();

        if ((c == 'r') || (c == 'R')){
            roi_corners.push_back(roi_corners[0]);
            roi_corners.erase(roi_corners.begin());
        }

        if ((c == 'i') || (c == 'I')){
            swap(roi_corners[0], roi_corners[1]);
            swap(roi_corners[2], roi_corners[3]);
        }
    }
    cvtColor(warped_image,warped_image,COLOR_BGR2GRAY);
    for(int i = 0; i<120; i++) medianBlur(warped_image,warped_image,3);
    for(int i = 0; i<120; i++) medianBlur(warped_image,warped_image,5);
    for(int i = 0; i<20; i++) medianBlur(warped_image,warped_image,7);

    namedWindow("AfterFilter", WINDOW_GUI_NORMAL);
    imshow("AfterFilter", warped_image);
    waitKey(0);
    return 0;
}

static void onMouse(int event, int x, int y, int, void*)
{
    const int drag_radius = 20; // 드래그할 코너의 크기를 조절하는 변수

    if (roi_corners.size() == 4)
    {
        for (int i = 0; i < 4; ++i)
        {
            if ((event == EVENT_LBUTTONDOWN) && (norm(roi_corners[i] - Point2f(x, y)) < drag_radius))
            {
                selected_corner_index = i;
                dragging = true;
            }
        }
    }
    else if (event == EVENT_LBUTTONDOWN)
    {
        roi_corners.push_back(Point2f((float)x, (float)y));
        validation_needed = true;
    }

    if (event == EVENT_LBUTTONUP)
    {
        dragging = false;
    }

    if ((event == EVENT_MOUSEMOVE) && dragging)
    {
        roi_corners[selected_corner_index].x = (float)x;
        roi_corners[selected_corner_index].y = (float)y;
        validation_needed = true;
    }
}