//
// Created by cc on 2019/11/25.
//

#ifndef HIST_KALMAN_MOT_OBJECT_IN_VIEW_H
#define HIST_KALMAN_MOT_OBJECT_IN_VIEW_H

#include <Eigen/Eigen>
#include <string>
#include <opencv2/opencv.hpp>


class ObjectInView{
public:
    std::string name_;
    double observed_time_;

    cv::Mat color_image_;

    /// Feature 1: label
    std::string label_;
    float label_confidence_;

    /// Feature 2: color histogram
    cv::Mat color_hist_;

    /// Feature 3: Position for Kalfilter
    Eigen::Vector3f position_;
};


#endif //HIST_KALMAN_MOT_OBJECT_IN_VIEW_H
