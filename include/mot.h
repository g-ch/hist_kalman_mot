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

class MOT{
public:
    bool use_feature_label;
    bool use_feature_hist;
    bool use_feature_kalman;

private:
    std::map<std::string, TargetInTrack*> objects_map;
    long int id_counter;

public:
    MOT():id_counter(0),
          use_feature_label(true),
          use_feature_hist(true),
          use_feature_kalman(true)
    {
        std::cout<<"Multiple Object Tracker created!"<<std::endl;
//        objects_map["aa"] = new TargetInTrack();
//        objects_map["aa"] -> name_ = "hh";
    }
    ~MOT(){;}

    void matchAndCreateObjects(std::vector<ObjectInView*> &objects_this){
        cv::imshow("ob", objects_this[0]->color_image_);
        cv::waitKey(1);

        /** Calculate histogram for every object image ROI **/
        if(use_feature_hist){
            for(const auto & object_i : objects_this){
                calHistHS(object_i->color_image_,  object_i->color_hist_);
            }
        }

        // Test matching by color histogram
//        static bool first_time = true;
//        static cv::MatND hist_base;
//        if(first_time){
//            first_time = false;
//            objects_this[0]->color_hist_.copyTo(hist_base);
//        }
//        else{
//            double similarity = costCalHistHS(hist_base,  objects_this[0]->color_hist_);
//            std::cout << similarity << std::endl;
//        }

        if(objects_map.size() > 0){
            /** Suppose only one dynamic object insight to test Kalman filter **/
            objects_map["Object0"]->updateTarget(objects_this[0]);

        }else{  // No object candidates, create directly.
            for(const auto & object_i : objects_this)
            {
                auto candidate_temp = new TargetInTrack("Object"+std::to_string(id_counter), object_i, 1.f);
                objects_map["Object"+std::to_string(id_counter)] = candidate_temp;
                id_counter ++;
            }
        }

    }

    void getObjectsStates();

private:
    void hungaryAllocate();

    void getHist();

    void calHistHS(cv::Mat &img, cv::MatND &hist){
        /* Calculate a two-dimension histogram from H channel and S channel (without V channel). )*/
        cv::Mat hsv_img;
        cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);

        cv::imshow("hsv", hsv_img);
        cv::waitKey(2);

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

    double costCalHistHS(const cv::MatND &hist1, const cv::MatND &hist2){
        /* hist1 and hist2 should be normalized and have the same dimension */
        // TODO: Test different comparison method. Note the range of the results would be different.
        return cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL); //cv::HISTCMP_BHATTACHARYYA
    }

    void costCalKalman();

    void costCalLabel();



};


#endif //HIST_KALMAN_MOT_MOT_CHG_H
