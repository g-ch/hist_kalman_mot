//
// Created by cc on 2019/11/23.
//

#ifndef HIST_KALMAN_MOT_TARGET_IN_TRACK_H
#define HIST_KALMAN_MOT_TARGET_IN_TRACK_H

#include <iostream>
#include <utility>
#include <object_in_view.h>

#define KF_PROCESS_NOISE 1e-3
#define KF_MEASUREMENT_NOISE 1e-1

class TargetInTrack{
public:
    std::string name_;
    double init_time_;
    int number_observed_times_;
    double last_observed_time_;

    /** Feature 1: label **/
    std::string label_;
    float label_confidence_;

    /** Feature 2: color histogram **/
    cv::MatND color_hist_;

    /** Feature 3: Kalman state **/
    Eigen::Vector3f state_position_;
    Eigen::Vector3f state_velocity_;
    float sigma_acc_;  // standard deviation for acceleration

private:
    Eigen::Vector3f observed_position_;
    std::vector<cv::KalmanFilter*> position_KF_;
    std::vector<cv::Mat> measurement_KF_;

public:
    TargetInTrack(std::string name, ObjectInView* object, float sigma_acc):
            name_(std::move(name)),
            init_time_(object->observed_time_),
            last_observed_time_(object->observed_time_),
            number_observed_times_(1),
            label_(object->label_),
            label_confidence_(object->label_confidence_),
            color_hist_(object->color_hist_),
            observed_position_(object->position_),
            sigma_acc_(sigma_acc){

        /** Define Kalman filters and related matrices, x y z are processed individually **/
        auto position_KF_x = new cv::KalmanFilter(2,1,0);
        auto position_KF_y = new cv::KalmanFilter(2,1,0);
        auto position_KF_z = new cv::KalmanFilter(2,1,0);
        position_KF_.push_back(position_KF_x);
        position_KF_.push_back(position_KF_y);
        position_KF_.push_back(position_KF_z);

        for(auto & KF_i : position_KF_)
        {
            KF_i->transitionMatrix = cv::Mat::eye(2, 2, CV_32F); // A, would change because of delt_t
            KF_i->transitionMatrix.at<float>(0,1) = 0.f;

            setIdentity(KF_i->measurementMatrix); // H
            setIdentity(KF_i->processNoiseCov, cv::Scalar::all(KF_PROCESS_NOISE)); // Q
            setIdentity(KF_i->measurementNoiseCov, cv::Scalar::all(KF_MEASUREMENT_NOISE)); // R
            setIdentity(KF_i->errorCovPost, cv::Scalar::all(1));
        }

        for(int i=0; i<3; i++){
            cv::Mat measurement_this_axis = cv::Mat::ones(1, 1, CV_32F) * observed_position_[i];
            measurement_KF_.push_back(measurement_this_axis);
            position_KF_[i]->statePost = (cv::Mat_<float>(2,1) << observed_position_[i], 0.f); // initialize states (position, velocity=0).
        }

        std::cout<<"Target: "<< name_ <<" created! z=" << observed_position_[2] << std::endl;
    }

//    TargetInTrack(){
//        std::cout << "new_target created!" <<std::endl;
//    };

    ~TargetInTrack(){
        ;
    }

    int updateTarget(ObjectInView* object){
//        std::cout << "position = (" << object->position_[0] << ", " << object->position_[1]<<", "<< object->position_[2]<<")"<<std::endl;
        double delt_t_ = object->observed_time_ - last_observed_time_;
        if(delt_t_ <= 0.f){
            std::cout << "Error: delt_t_ can not be negative! Please check the time stamp!" << std::endl;
        }

        last_observed_time_ = object->observed_time_;
        updateKalmanState(delt_t_, object->position_);
        return 1;
    }

private:
    void updateLabelConfidence(){
        ;
    }

    void updateHist(){
        ;
    }

    void updateKalmanState(double &time_interval, Eigen::Vector3f &observed_position){
        observed_position_ = observed_position;

        for(int i=0; i<3; i++){
            /** Prediction **/
            position_KF_[i]->transitionMatrix.at<float>(0,1) = time_interval;
            position_KF_[i]->predict();

            /** Correction **/
            measurement_KF_[i] = cv::Mat::ones(1, 1, CV_32F) * observed_position_[i];
            position_KF_[i]->correct(measurement_KF_[i]);
            state_position_[i] = position_KF_[i]->statePost.at<float>(0);
            state_velocity_[i] = position_KF_[i]->statePost.at<float>(1);
        }
        std::cout << "time interval = "<< position_KF_[0]->transitionMatrix.at<float>(0,1) << std::endl;
        std::cout << "position observed = (" << observed_position[0] << ", " << observed_position[1]<<", "<< observed_position[2]<<")"<<std::endl;
        std::cout << "position corrected = (" << state_position_[0] << ", " << state_position_[1]<<", "<< state_position_[2]<<")"<<std::endl;
        std::cout << "velocity estimated = (" << state_velocity_[0] << ", " << state_velocity_[1]<<", "<< state_velocity_[2]<<")"<<std::endl;
    }
};

#endif //HIST_KALMAN_MOT_TARGET_IN_TRACK_H
