//
// Created by cc on 2019/11/23.
//

#ifndef HIST_KALMAN_MOT_MOT_CHG_H
#define HIST_KALMAN_MOT_MOT_CHG_H

#include <target_in_track.h>
#include <map>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "munkres.h"

class MOT{
public:
    bool use_feature_label;
    bool use_feature_hist;
    bool use_feature_kalman;

private:
    std::map<std::string, TargetInTrack*> objects_map;
    std::map<std::string, TargetInTrack*>::iterator map_iterator;
    Munkres<float> munkres_solver;
    long int id_counter;

public:
    MOT():id_counter(0),
          use_feature_label(true),
          use_feature_hist(true),
          use_feature_kalman(true)
    {
        std::cout<<"Multiple Object Tracker created!"<<std::endl;
    }

    ~MOT(){;}

    void matchAndCreateObjects(std::vector<ObjectInView*> &objects_this){
        cv::imshow("ob", objects_this[0]->color_image_);
        cv::waitKey(1);

        cv::RNG rng(cvGetTickCount()); //random_color_generator

        /** Calculate histogram for every object image ROI if use color histogram feature**/
        if(use_feature_hist){
            for(const auto & object_i : objects_this){
                calHistHS(object_i->color_image_,  object_i->color_hist_);
            }
        }

        /** Match and update **/
        if(objects_map.size() > 0){

            /** Allocate by Kuhn–Munkres algorithm **/
            int num_objects_in_view = objects_this.size();
            int num_objects_in_storage = objects_map.size();

            Matrix<float> matrix_cost(num_objects_in_view, num_objects_in_storage); //This is a Matrix defined in munkres.h
            Matrix<float> matrix_gate(num_objects_in_view, num_objects_in_storage);

            static const float similarity_position_gate = 0.f;  //TODO: Tune the gate parameters here
            static const float similarity_histogram_gate = 0.1f;
            static const float similarity_label_gate = 0.2f;

            for(int row=0; row<num_objects_in_view; row++)
            {
                map_iterator = objects_map.begin();
                for(int col=0; col<num_objects_in_storage; col++)  // col is one-to-one corresponding to map_iterator.
                {
                    float similarity_position = 0.f, similarity_histogram = 0.f, similarity_label = 0.f;
                    /// Position
                    similarity_position = similarityPositionDist(objects_this[row], map_iterator->second);
                    if(similarity_position < similarity_position_gate){
                        matrix_gate(row, col) = 0.f;
                    }else{
                        matrix_gate(row, col) = 1.f;
                    }

                    /// Histogram
                    if(use_feature_hist) {
                        similarity_histogram = similarityCalHistHS(objects_this[row], map_iterator->second);
                        if(similarity_histogram < similarity_histogram_gate){
                            matrix_gate(row, col) = 0.f;
                        }
                    }

                    /// Label
                    if(use_feature_label) {
                        similarity_label = similarityCalLabel(objects_this[row], map_iterator->second);
                        if(similarity_label < similarity_label_gate){
                            matrix_gate(row, col) = 0.f;
                        }
                    }

                    /// Combinition
                    static const float lamda_position = 0.4; //TODO: Tune the parameters
                    static const float lamda_hist = 0.3;
                    static const float lamda_label = 1.f - lamda_position - lamda_hist;

                    matrix_cost(row,col) = 1.f - (float)(lamda_position * similarity_position + lamda_hist * similarity_histogram
                                            + lamda_label * similarity_label);
                    map_iterator ++;
                }
            }

            munkres_solver.solve(matrix_cost);

            std::vector<std::string> allocation_result;
            static std::string label_no_match = "nothing";

            for(int row=0; row<num_objects_in_view; row++)
            {
                map_iterator = objects_map.begin();
                bool found_match = false;
                for(int col=0; col<num_objects_in_storage; col++)  // col is one-to-one corresponding to map_iterator.
                {
                    if(matrix_cost(row, col) == 0.f && matrix_gate(row, col) != 0.f){  // Found a match
                        allocation_result.push_back(map_iterator->first);
                        found_match = true;
                        break;
                    }
                    map_iterator ++;
                }

                if(!found_match){
                    allocation_result.push_back(label_no_match);
                }
            }

//            if(allocation_result.size() != objects_this.size()){ std::cout << "Code error !!!!!!!!!!!! chg"<< std::endl; return;}

            for(auto & result_i : allocation_result){

            }

            /** Update objects in storage or create new objects according to allocation result **/
            for(int i=0; i<allocation_result.size(); i++)
            {
                if(allocation_result[i]!=label_no_match){  /// Found a match
                    objects_map[allocation_result[i]]->updateTarget(objects_this[i]);
                    objects_this[i]->color_to_show_ = objects_map[allocation_result[i]]->color_to_show_;  /// Update name and color
                    objects_this[i]->name_ = objects_map[allocation_result[i]]->name_;

                }else{  /// Found no match. Create a new one.

                    cv::Scalar random_color = cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
                    objects_this[i]->color_to_show_ = random_color;
                    objects_this[i]->name_ = objects_this[i]->label_ + std::to_string(id_counter);

                    auto candidate_temp = new TargetInTrack(objects_this[i]->label_ + std::to_string(id_counter), objects_this[i]);
                    objects_map[objects_this[i]->label_ + std::to_string(id_counter)] = candidate_temp;
                    id_counter ++;
                }
            }

        }else{  /** No object candidates in storage. Create new candidates directly. **/
            for(const auto & object_i : objects_this)
            {
                cv::Scalar random_color = cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
                object_i->color_to_show_ = random_color;
                object_i->name_ = object_i->label_ + std::to_string(id_counter);

                auto candidate_temp = new TargetInTrack(object_i->label_ + std::to_string(id_counter), object_i);
                objects_map[object_i->label_ + std::to_string(id_counter)] = candidate_temp;
                id_counter ++;
            }
        }

    }

    void getObjectsStates();

private:
    float similarityCalHistHS(ObjectInView* object, TargetInTrack* target_stored){
        /** Similarity of color histogram in H and S channel. [0,1]
         * hist1 and hist2 should be normalized and have the same dimension **/
        // TODO: Test different comparison method. Note the range of the results would be different.
        return cv::compareHist(object->color_hist_, target_stored->color_hist_, cv::HISTCMP_CORREL); //cv::HISTCMP_BHATTACHARYYA
    }

    float similarityPositionDist(ObjectInView* object, TargetInTrack* target_stored){
        /** Similarity of position judging by Mahalanobis Distance. [0,1]**/
        float delt_t = object->observed_time_ - target_stored->last_observed_time_;
        if(delt_t <= 0.f){
            std::cout << "Error: delt_t should be position!" << std::endl;
            return 0.f;
        }else{
            return exp(-target_stored->futurePassProbabilityMahalanobis(delt_t, object->position_, 0.5));  // TODO: tune the cov_delt_t_limitation parameter
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
        /* Calculate a two-dimension histogram from H channel and S channel (without V channel). )*/
        cv::Mat hsv_img;
        cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);

        const int h_bins = 60;
        const int s_bins = 128;
        int hist_size[] = {h_bins, s_bins};
        float h_ranges[] = {0, 180};
        float s_ranges[] = {0, 256};
        const float* ranges[] = {h_ranges, s_ranges};
        int channels[] = {0, 1};

        cv::calcHist(&hsv_img, 1, channels, cv::Mat(), hist, 2, hist_size, ranges, true, false);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }

};


#endif //HIST_KALMAN_MOT_MOT_CHG_H
