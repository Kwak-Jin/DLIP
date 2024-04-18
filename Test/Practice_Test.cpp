/** @about DLIP Mid Term
*  @Author Jin Kwak 21900031
 *  @created 2024.04.15
 */

#include <DLIP.hpp>

cv::Mat src,dst,dst_prac,src_prac;

int main(){
	src =  cv::imread("../Image/track_gray.jpg",1);
	if (!src.data)
	{
		return -1;
	}
	cv::cvtColor(src, src_prac,  cv::COLOR_BGR2GRAY);
	cv::cornerHarris(src_prac,dst_prac,2,5,0.05);


	std::cout<<src.size()<<std::endl;
	cv::Rect roi( 120, 50, 493-2*120, 349 );
	src = src(roi);

	cv::medianBlur(src,src,3);
	cv::medianBlur(src,src,3);
	cv::medianBlur(src,src,3);
	dst.create(src.size(), src.type());
	cv::Canny(src, dst, 30,200, 3);

	std::vector<cv::Vec2f> lines;
	cv::HoughLines(dst, lines, 1, CV_PI / 180, 141, 0, 0);
	double angle_control;
	double _lines;
	// Draw the detected lines
	for (cv::Vec2f line: lines){
		float rho = line[0], theta = line[1];
		angle_control += theta;
		_lines++;
		std::cout<<"THETA = " << theta<<std::endl;
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cv::line(src, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}
	double steerCMD = angle_control/_lines;
	std::cout<<"Steer Command = " << steerCMD<<std::endl;
	cv::imshow("check", src);
	cv::imshow("Corner_src",src_prac);
	cv::imshow("Corner",dst_prac);
	cv::waitKey(0);
	return 0;
}