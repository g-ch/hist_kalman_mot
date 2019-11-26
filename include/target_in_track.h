//
// Created by cc on 2019/11/23.
//

#ifndef HIST_KALMAN_MOT_TARGET_IN_TRACK_H
#define HIST_KALMAN_MOT_TARGET_IN_TRACK_H

#include <iostream>
#include <utility>
#include <object_in_view.h>

class TargetInTrack : public ObjectInView{
public:
    double init_time_;
    int number_observed_times_;

    /// Feature 1: label (defined in parent class)

    /// Feature 2: color histogram (defined in parent class)

    /// Feature 3: Kalman state
    cv::Mat corrected_kalman_state_;
    float sigma_acc_;  // standard deviation for acceleration

public:
//    TargetInTrack(std::string name, double init_time, std::string label, float label_confidence,
//                  cv::Mat color_hist, cv::Mat init_kalman_state, float sigma_acc):
//            name_(std::move(name)),
//            init_time_(init_time),
//            last_observed_time_(init_time),
//            observed_times_(1),
//            label_(std::move(label)),
//            label_confidence_(label_confidence),
//            color_hist_(std::move(color_hist)),
//            corrected_kalman_state_(std::move(init_kalman_state)),
//            sigma_acc_(sigma_acc){
//        ;
//    }

    TargetInTrack(){
        std::cout << "new_target created!" <<std::endl;
    };

    ~TargetInTrack(){
        ;
    }

//    void getLabel(std::string &target_label, float &target_label_confidence){
//        ;
//    }
//
//    void getHist(cv::Mat &target_color_hist){
//        ;
//    }
//
//    void getKalmanState(cv::Mat &target_kalman_state, cv::Mat &target_state_var, float &target_sigma_acc){
//        ;
//    }

private:
    void updateLabelConfidence(){
        ;
    }

    void updateKalmanState(){
        ;
    }

    int updateTarget(){
        return 0;
    }

};

#endif //HIST_KALMAN_MOT_TARGET_IN_TRACK_H
