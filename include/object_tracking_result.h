//
// Created by cc on 2019/11/29.
//

#ifndef HIST_KALMAN_MOT_OBJECT_TRACKING_RESULT_H
#define HIST_KALMAN_MOT_OBJECT_TRACKING_RESULT_H

class ObjectTrackingResult{
public:
    std::string name_;
    double last_observed_time_;
    std::string label_;

    Eigen::Vector3f position_;
    Eigen::Vector3f velocity_;
    double sigma_;

public:
    ObjectTrackingResult(){
    }
    ~ObjectTrackingResult(){;}
};

#endif //HIST_KALMAN_MOT_OBJECT_TRACKING_RESULT_H
