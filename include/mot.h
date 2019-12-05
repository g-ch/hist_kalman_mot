//
// Created by cc on 2019/11/23.
//

#ifndef HIST_KALMAN_MOT_MOT_CHG_H
#define HIST_KALMAN_MOT_MOT_CHG_H

#include <target_in_track.h>
#include <object_tracking_result.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "munkres.h"

class MOT{
private:
    std::map<std::string, TargetInTrack*> objects_map_;
    std::map<std::string, TargetInTrack*>::iterator map_iterator_;
    long int id_counter_;

    std::map<std::string, float> sigma_acc_map_;

    double time_out_threshold_;
    double position_out_threshold_;

    bool use_feature_label_;
    bool use_feature_hist_;
    bool use_feature_kalman_;

    float similarity_position_gate_;
    float similarity_histogram_gate_;
    float similarity_label_gate_;

    float lamda_position_;
    float lamda_hist_;
    float lamda_label_;

public:
    MOT():id_counter_(0),
          use_feature_label_(true),
          use_feature_hist_(true),
          use_feature_kalman_(true),
          time_out_threshold_(10.0),
          position_out_threshold_(10.0),
          similarity_position_gate_(0.1f),
          similarity_histogram_gate_(0.1f),
          similarity_label_gate_(0.3f),
          lamda_position_(0.3),
          lamda_hist_(0.3),
          lamda_label_(0.4)
    {
        /** Set a default sigma_acc_map **/
        sigma_acc_map_["person"] = 2.f;
        sigma_acc_map_["car"] = 3.f;

        std::cout<<"Multiple Object Tracker created!"<<std::endl;
    }

    ~MOT(){;}

    void setCandidateOutThreshold(double time_threshold, double position_threshold){
        /** This function is to set thresholds to remove useless object candidates in storage.
         * @Parameter time_threshold: If one object candidate in storage has not been updated for time_threshold seconds, it will be deleted.
         * @Parameter position_threshold: If the estimated distance between the camera and one object candidate is larger than position_threshold, the candidate will be deleted.
         * **/
        time_out_threshold_ = time_threshold;
        position_out_threshold_ = position_threshold;
    }

    void setSimilarityGateValues(float similarity_position_gate, float similarity_histogram_gate, float similarity_label_gate){
        /** This function is to set gate values for matching. If a new object's similarity towards a candidate is smaller than the gate, they would not match.
         * If in Function setMultiCueCoefficients() the weight coefficient for one cue is set to be zero, the corresponding gate won't work and can be any value.
         * @Parameter similarity_position_gate: [0, 1]
         * @Parameter similarity_histogram_gate: [0, 1]
         * @Parameter similarity_label_gate: [0, 1]
         * **/
        similarity_position_gate_ = similarity_position_gate;
        similarity_histogram_gate_ = similarity_histogram_gate;
        similarity_label_gate_ = similarity_label_gate;
    }

    void setMultiCueCoefficients(float lamda_position, float lamda_hist, float lamda_label){
        /** This function is to set weight coefficients in matching with multiple cues.
         * The cues are position, color histogram and label.
         * Note: the coefficients must satisfy: lamda_position + lamda_hist + lamda_label = 1.f
         * If you don't want to use a cue, set the corresponding weight lamda_XX to 0.f. Cue position must be used!!
         * @Parameter lamda_position: (0,1]
         * @Parameter lamda_hist: [0,1)
         * @Parameter lamda_label: [0,1)
         * **/

        if(lamda_position < 0.f || lamda_hist < 0.f || lamda_label < 0.f){
            std::cout << "Error: lamda should not be negative! Use default." << std::endl;
        }

        if(lamda_position + lamda_hist + lamda_label != 1.f){
            std::cout << "Error: the sum of the coefficients should be 1.f! Use default." << std::endl;
        }

        lamda_position_ = lamda_position;
        lamda_hist_ = lamda_hist;
        lamda_label_ = lamda_label;

        if(lamda_hist_ == 0.f) use_feature_hist_ = false;
        if(lamda_label_ == 0.f) use_feature_hist_ = false;
    }

    void setAccelerationVarianceMap(std::map<std::string, float> &sigma_acc_map){
        /** This function is to set a map <label, acceleration variance> for position prediction.
         * The motion model for objects are uniform velocity model, where acceleration obeys N(0, sqrt(variance)).
         * If this function is not performed or a vacant map is given. Default variance 1.0 will be used.
         * @Parameter sigma_acc_map: map <label, acceleration variance>
         * **/
        sigma_acc_map_.clear();
        for(auto iterator_ = sigma_acc_map.begin(); iterator_ != sigma_acc_map.end(); iterator_ ++){
            sigma_acc_map_[iterator_->first] = iterator_->second;
        }
    }

    void matchAndCreateObjects(std::vector<ObjectInView*> &objects_this, Eigen::Vector3f &camera_position){
        /** This function is to update tracking results with newly detected objects.
         * @Parameter objects_this: a vector to store all the dynmaic objects detected in view.
         * @Parameter camera_position: the position of the camera required for object candidate delete. If camera is fixed, use zeros. The position of the camera and the objects should be measured in a same global coordinate
         * **/

        /** Check if empty **/
        if(objects_this.empty()){
            std::cout << "Warning: No objects in this vector. Don't send an empty vector to the tracker!" << std::endl;
            return;
        }

        /** Calculate histogram for every object image ROI if use color histogram feature**/
        if(use_feature_hist_){
            for(auto & object_i : objects_this){
                calHistHS(object_i->color_image_,  object_i->color_hist_);
            }
        }

        /** Match and update **/
        cv::RNG rng(cvGetTickCount()); //random_color_generator
        if(objects_map_.size() > 0){

            /** Calculate cost matrix **/
            int num_objects_in_view = objects_this.size();
            int num_objects_in_storage = objects_map_.size();

            Matrix<float> matrix_cost(num_objects_in_view, num_objects_in_storage); //This is a Matrix defined in munkres.h
            Matrix<float> matrix_gate(num_objects_in_view, num_objects_in_storage);

            for(int row=0; row<num_objects_in_view; row++)
            {
                map_iterator_ = objects_map_.begin();

                for(int col=0; col<num_objects_in_storage; col++)  // col is one-to-one corresponding to map_iterator.
                {
                    float similarity_position = 0.f, similarity_histogram = 0.f, similarity_label = 0.f;
                    /// Position
                    similarity_position = similarityPositionDist(objects_this[row], map_iterator_->second);
                    if(similarity_position < similarity_position_gate_){
                        matrix_gate(row, col) = 0.f;
                    }else{
                        matrix_gate(row, col) = 1.f;
                    }
                    /// Histogram
                    if(use_feature_hist_) {
                        similarity_histogram = similarityCalHistHS(objects_this[row], map_iterator_->second);
                        if(similarity_histogram < similarity_histogram_gate_){
                            matrix_gate(row, col) = 0.f;
                        }
                    }
                    /// Label
                    if(use_feature_label_) {
                        similarity_label = similarityCalLabel(objects_this[row], map_iterator_->second);
                        if(similarity_label < similarity_label_gate_){
                            matrix_gate(row, col) = 0.f;
                        }
                    }
                    /// Combinition
                    matrix_cost(row,col) = 1.f - (float)(lamda_position_ * similarity_position + lamda_hist_ * similarity_histogram
                                                         + lamda_label_ * similarity_label);
                    map_iterator_ ++;
                }
            }

            /** Allocate by Kuhn–Munkres algorithm **/
            Munkres<float> munkres_solver;
            munkres_solver.solve(matrix_cost);

            std::vector<std::string> allocation_result;
            static std::string label_no_match = "nothing";

            for(int row=0; row<num_objects_in_view; row++)
            {
                map_iterator_ = objects_map_.begin();
                bool found_match = false;
                for(int col=0; col<num_objects_in_storage; col++)  // col is one-to-one corresponding to map_iterator.
                {
                    if(matrix_cost(row, col) == 0.f){  // Found a match
                        if(matrix_gate(row, col) > 0.f){
                            allocation_result.push_back(map_iterator_->first);
                            found_match = true;
                            break;
                        }else{
                            std::cout << "Gate failed for this allocation!" << std::endl;
                        }

                    }
                    map_iterator_ ++;
                }

                if(!found_match){
                    allocation_result.push_back(label_no_match);
                }
            }

            /** Update objects in storage according to allocation result **/
            for(int i=0; i<allocation_result.size(); i++)  //allocation_result.size() is the same as objects_this.size()
            {
                if(allocation_result[i]!=label_no_match){  /// Found a match
                    objects_map_[allocation_result[i]]->updateTarget(objects_this[i]);

                    objects_this[i]->color_to_show_ = objects_map_[allocation_result[i]]->color_to_show_;  /// Update name and color
                    objects_this[i]->name_ = objects_map_[allocation_result[i]]->name_;
                }
            }

            /** Create new objects according to allocation result **/
            for(int i=0; i<allocation_result.size(); i++)
            {
                /// Found no match. Create a new one. Note new objects should not be used to match objects in this frame. So this should be in another loop.
                if(allocation_result[i]==label_no_match){
                    cv::Scalar random_color = cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
                    objects_this[i]->color_to_show_ = random_color;
                    objects_this[i]->name_ = objects_this[i]->label_ + std::to_string(id_counter_);

                    auto candidate_temp = new TargetInTrack(objects_this[i]->label_ + std::to_string(id_counter_), objects_this[i], sigma_acc_map_);
                    objects_map_[objects_this[i]->label_ + std::to_string(id_counter_)] = candidate_temp;  // Create new
                    id_counter_ ++;
                }
            }

        }else{  /** No object candidates in storage. Create new candidates directly. **/
            for(const auto & object_i : objects_this)
            {
                cv::Scalar random_color = cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
                object_i->color_to_show_ = random_color;
                object_i->name_ = object_i->label_ + std::to_string(id_counter_);

                auto candidate_temp = new TargetInTrack(object_i->label_ + std::to_string(id_counter_), object_i, sigma_acc_map_);
                objects_map_[object_i->label_ + std::to_string(id_counter_)] = candidate_temp; // Create new
                id_counter_ ++;
            }
        }
    }

    int getObjectsStates(std::vector<ObjectTrackingResult*> &result){
        /** This function fetches the properties of objects in current tracking result and return objects number
         * @Paramter result: a vector to store the result
         * @Return number of the tracked objects
         * **/
        int counter = 0;
        for(map_iterator_ = objects_map_.begin(); map_iterator_!= objects_map_.end(); map_iterator_++)
        {
            ObjectTrackingResult* ob_temp = new ObjectTrackingResult();
            ob_temp->name_ = map_iterator_->second->name_;
            ob_temp->label_ = map_iterator_->second->label_;
            ob_temp->last_observed_time_ = map_iterator_->second->last_observed_time_;
            ob_temp->position_ = map_iterator_->second->state_position_;
            ob_temp->velocity_ = map_iterator_->second->state_velocity_;
            ob_temp->sigma_ = map_iterator_->second->sigma_acc_;
            result.push_back(ob_temp);
            counter ++;
        }
        return counter;
    }

    void checkUselessObjects(Eigen::Vector3f &camera_position,  double &time_stamp){
        /** Delete object candidates in storage which are time-out or position-out **/
        deleteUselessCandidates(time_stamp, camera_position);
    }

private:
    float similarityCalHistHS(ObjectInView* object, TargetInTrack* target_stored){
        /** Similarity of color histogram in H and S channel. [0,1]
         * hist1 and hist2 should be normalized and have the same dimension **/
        // TODO: Test different comparison method. Note the range of the results would be different.
        return cv::compareHist(object->color_hist_, target_stored->color_hist_, cv::HISTCMP_CORREL); //cv::HISTCMP_BHATTACHARYYA
    }

    float similarityPositionDist(ObjectInView* object, TargetInTrack* target_stored){
        /** Similarity of position judging by Mahalanobis Distance. [0,1]**/
        double delt_t = object->observed_time_ - target_stored->last_observed_time_;
        if(delt_t <= 0.f){
            std::cout << "Error: delt_t should be positive!" << std::endl;
            return 0.f;
        }else{
            return exp(-target_stored->futurePassProbabilityMahalanobis(delt_t, object->position_, 0.5) * 0.1);  // TODO: tune the cov_delt_t_limitation parameter
        }
    }

    float similarityCalLabel(ObjectInView* object, TargetInTrack* target_stored){
        /** Similarity of label. [0,1] **/
        if(object->label_ == target_stored->label_){
            return object->label_confidence_ * target_stored->label_confidence_;
        }else{
            return object->label_confidence_ * (1.f - target_stored->label_confidence_) * 0.5f;  // TODO: tune the punishment parameter
        }
    }

    void calHistHS(cv::Mat &img, cv::MatND &hist){
        /** Calculate a two-dimension histogram from H channel and S channel (without V channel). ) **/
        cv::Mat hsv_img;
        cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);

        const int h_bins = 60;   // TODO: tune the histogram parameter
        const int s_bins = 128;
        int hist_size[] = {h_bins, s_bins};
        float h_ranges[] = {0, 180};
        float s_ranges[] = {2, 256};  ///NOTE: pixels with saturation lower than 2 (white) will not be included.
        const float* ranges[] = {h_ranges, s_ranges};
        int channels[] = {0, 1};

        cv::calcHist(&hsv_img, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }

    int deleteUselessCandidates(const double &time_now, const Eigen::Vector3f &camera_position_now){
        /** Delete object candidates in storage that are time-out or position-out. **/
        int deleted_candidate_num = 0;

        for(map_iterator_ = objects_map_.begin(); map_iterator_!= objects_map_.end(); )
        {
            double time_from_last_update = time_now - map_iterator_->second->last_observed_time_;

            Eigen::Vector3f distance_vector = map_iterator_->second->futurePositionMostLikely(time_now) - camera_position_now;
            double square_distance_predict_to_camera = distance_vector.squaredNorm();

            if(time_from_last_update > time_out_threshold_ || square_distance_predict_to_camera > position_out_threshold_*position_out_threshold_){
                std::cout << "Delete " << map_iterator_->first << std::endl;
                objects_map_.erase(map_iterator_++);
                deleted_candidate_num ++;
            }else{
                map_iterator_++;
            }
        }

        return  deleted_candidate_num;
    }


};


#endif //HIST_KALMAN_MOT_MOT_CHG_H
