//
// Created by cc on 2019/11/25.
//

#ifndef HIST_KALMAN_MOT_OBJECT_IN_VIEW_H
#define HIST_KALMAN_MOT_OBJECT_IN_VIEW_H

#include <Eigen/Eigen>
#include <string>
#include <opencv2/opencv.hpp>
#include <math.h>

class ObjectInView{
public:
    std::string name_;
    double observed_time_;
    cv::Mat color_image_;

    cv::Rect img_pixel_rect_; //This is just for visualization
    cv::Scalar color_to_show_;  //This is just for visualization

    /// Feature 1: label
    std::string label_;
    float label_confidence_;

    /// Feature 2: color histogram
    cv::MatND color_hist_;

    /// Feature 3: Position for Kalfilter
    Eigen::Vector3f position_;


public:
    ObjectInView(){
        color_to_show_ = cv::Scalar(255, 255, 255);
    }

    ~ObjectInView(){;}
};

/** Mahalanobis Distance calculation **/
float calMahalanobisDistance3D(Eigen::Vector3f &point, Eigen::Vector3f &distribution_avg, Eigen::Matrix3f &distribution_cov)
{
    Eigen::Vector3f delt = point - distribution_avg;
    return sqrt(delt.transpose() * distribution_cov.inverse() * delt);
}

#endif //HIST_KALMAN_MOT_OBJECT_IN_VIEW_H
