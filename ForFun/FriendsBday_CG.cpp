/** @creator Jin Kwak
 * @brief Photoshop friends face
 * @created 24.04.23
 * @birthday 24.04.24
*/

#include <opencv.hpp>
#include <iostream>

cv::Mat Tgeon1, Tgeon2, Tgeon3, Tg1, Tg2, Tg3;

int main(){
 Tgeon1 = cv::imread("Tgeon1.jpg");
 Tgeon2 = cv::imread("Tgeon2.jpg");
 Tgeon3 = cv::imread("Tgeon3.jpg");
 // cv::cvtColor(Tgeon1,Tgeon1, cv::COLOR_BGR2GRAY);
 // for(int idx = 0; idx <100 ; idx++) cv::medianBlur(Tgeon1,Tgeon1,7);
 // for(int idx = 0; idx <20 ; idx++) cv::GaussianBlur(Tgeon1,Tgeon1,cv::Size(3,3),0);
 // cv::equalizeHist(Tgeon1,Tgeon1);
 // cv::cvtColor(Tgeon1,Tg1, cv::COLOR_HSV2BGR);
 // cv::imshow("Tgeon1",Tg1);

 cv::cvtColor(Tgeon3,Tg3,cv::COLOR_BGR2GRAY);
 std::vector<cv::Vec3f> circles;
 cv::HoughCircles(Tg3, circles,3,1,Tg3.rows / 4, 30, 15);
 for (size_t i = 0; i < circles.size(); i++){
  cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
  int radius = cvRound(circles[i][2]);

  /* draw the circle outline */
  circle(Tgeon3, center, radius, cv::Scalar(255, 0, 255), 2, 8, 0);
 }
 cv::imshow("Tg3 Mango", Tgeon3);
 cv::waitKey(0);
 return 0;
}