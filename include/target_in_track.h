//
// Created by cc on 2019/11/23.
//

#ifndef HIST_KALMAN_MOT_TARGET_IN_TRACK_H
#define HIST_KALMAN_MOT_TARGET_IN_TRACK_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <utility>

using namespace std;

class TargetInTrack{
public:
    string name_;
private:
    double init_time_;
    double last_observed_time_;
    int observed_times_;

    /// Feature 1: label
    string label_;
    float label_confidence_;

    /// Feature 2: color histogram
    cv::MatND color_hist_;

    /// Feature 3: Kalman state
    cv::Mat corrected_kalman_state_;
    float sigma_acc_;  // standard deviation for acceleration

public:
//    TargetInTrack(string name, double init_time, string label, float label_confidence,
//                  cv::MatND color_hist, cv::Mat init_kalman_state, float sigma_acc):
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
        cout << "new_target created!" <<endl;
    };

    ~TargetInTrack(){
        ;
    }

    void get_label(string &target_label, float &target_label_confidence){
        ;
    }

    void get_hist(cv::MatND &target_color_hist){
        ;
    }

    void get_kalman_state(cv::Mat &target_kalman_state, cv::Mat &target_state_var, float &target_sigma_acc){
        ;
    }

private:
    void update_label_confidence(){
        ;
    }

    void update_kalman_state(){
        ;
    }

    int update_target(){
        return 0;
    }

};

#endif //HIST_KALMAN_MOT_TARGET_IN_TRACK_H
