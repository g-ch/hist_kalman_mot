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
    cv::Scalar color_to_show_;

    /** Feature 1: label **/
    std::string label_;
    float label_confidence_;

    /** Feature 2: color histogram **/
    cv::MatND color_hist_;

    /** Feature 3: Kalman state **/
    Eigen::Vector3f state_position_;
    Eigen::Vector3f state_velocity_;


private:
    Eigen::Vector3f observed_position_;
    std::vector<cv::KalmanFilter*> position_KF_;
    std::vector<cv::Mat> measurement_KF_;
    std::map<std::string, float> sigma_acc_map_;
    float sigma_acc_;  // Variance for acceleration

public:
    TargetInTrack(std::string name, ObjectInView* object):
            name_(std::move(name)),
            init_time_(object->observed_time_),
            last_observed_time_(object->observed_time_),
            number_observed_times_(1),
            label_(object->label_),
            label_confidence_(object->label_confidence_),
            color_hist_(object->color_hist_),
            observed_position_(object->position_)
            {

        /** Initialize acceleration sigma map, acc is a Gaussian distribution (0, sigma) **/
        // TODO: add more sigma_acc_ categories
        sigma_acc_map_["person"] = 1.f;
        sigma_acc_map_["car"] = 3.f;
        resetSigmaAcc();

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

        /** Other initialization **/
        state_position_ = observed_position_;
        state_velocity_ << 0.f, 0.f, 0.f;
        color_to_show_ = object->color_to_show_;

        std::cout<<"New target: "<< name_ <<" created! z=" << observed_position_[2] << std::endl;
    }

    ~TargetInTrack(){
        ;
    }

    int updateTarget(ObjectInView* object){
        double delt_t_ = object->observed_time_ - last_observed_time_;
        if(delt_t_ <= 0.f){
            std::cout << "Error: delt_t_ can not be negative! Please check the time stamp! delt_t_=" << delt_t_ << std::endl;
            return 0;
        }

        updateHist(object->color_hist_);
        updateKalmanState(delt_t_, object->position_);
        updateLabelConfidence(object->label_, object->label_confidence_);

        last_observed_time_ = object->observed_time_;
        number_observed_times_ ++;

        return 1;
    }

    float futurePassProbabilityMahalanobis(double delt_t, Eigen::Vector3f position, double cov_delt_t_limitation){
        /** Paras @ delt_t: time interval from now to the experted prediction time
         * Paras @ position: to predict whether the object will pass the position at the experted prediction time.
         * Paras @ cov_delt_t_limitation: limit the delt_t to limit distribution_cov. In case the covariance is too large that passing every point is possible.
         * **/
        Eigen::Vector3f predicted_position_center = state_velocity_ * delt_t + state_position_;  /// Predict most likely position position. Constant velocity model
//        std::cout << "position predicted = (" << predicted_position_center[0] << ", " << predicted_position_center[1] <<", "<< predicted_position_center[2]<<")"<<std::endl;
//        std::cout << "position cal point = (" << position[0] << ", " << position[1]<<", "<< position[2]<<")"<<std::endl;

        double cov_delt_t = std::min(cov_delt_t_limitation, delt_t);
        Eigen::Matrix3f distribution_cov = Eigen::Matrix3f::Identity() * sigma_acc_ * 0.25 * cov_delt_t * cov_delt_t;

        return (float)calMahalanobisDistance3D(position, predicted_position_center, distribution_cov);
    }

private:
    void updateLabelConfidence(std::string &label, float &confidence){
        /** Start from Bayes law and end up with increase or decrease a constant value in log-wise form. **/
        static const float decrease_log_p = 0.1f;
        static const float increase_log_p = 0.1f;
        static const float label_change_threshold = 0.2f;

        if(label == label_){
            label_confidence_ += increase_log_p;
        }else{
            label_confidence_ -= decrease_log_p;
        }

        if(label_confidence_ <= label_change_threshold){  /// Change label and name if confidence is too low.
            label_ = label;
            name_ = label_ + "_" + name_;  /// add a new label on the name
            label_confidence_ = confidence;
            resetSigmaAcc();
        }

        label_confidence_ = std::max(label_confidence_, 0.f);
        label_confidence_ = std::min(label_confidence_, 1.f);
//        std::cout<<label_<<": confidence = "<<label_confidence_<<std::endl;
    }

    void updateHist(cv::MatND& color_hist){
        /** Only keep the latest histogram for now. **/
        //TODO: add key histograms
        color_hist_ = color_hist;
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
//        std::cout << "time interval = "<< position_KF_[0]->transitionMatrix.at<float>(0,1) << std::endl;
////        std::cout << "position observed = (" << observed_position[0] << ", " << observed_position[1]<<", "<< observed_position[2]<<")"<<std::endl;
//        std::cout << "position corrected = (" << state_position_[0] << ", " << state_position_[1]<<", "<< state_position_[2]<<")"<<std::endl;
//        std::cout << "velocity estimated = (" << state_velocity_[0] << ", " << state_velocity_[1]<<", "<< state_velocity_[2]<<")"<<std::endl;
    }

    int resetSigmaAcc(){
        /** Reset acceleration variance when a new object comes or when the label of an object changes.
         * Return 1 if variance for this label is in the sigma_acc_map_. Otherwise use default_sigma_acc and return 0 **/
        static float default_sigma_acc = 1.f;
        if(sigma_acc_map_.count(name_) > 0){
            sigma_acc_ = sigma_acc_map_[name_];
            return 1;
        }else{
            sigma_acc_ = default_sigma_acc;  // if no corresponding sigma_acc_ found, use default sigma_acc_.
            return 0;
        }
    }
};

#endif //HIST_KALMAN_MOT_TARGET_IN_TRACK_H
