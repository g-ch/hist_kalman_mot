//
// Created by cc on 2019/11/23.
//

#ifndef HIST_KALMAN_MOT_MOT_ROS_H
#define HIST_KALMAN_MOT_MOT_ROS_H

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
        objects_map["aa"] = new TargetInTrack();
        objects_map["aa"] -> name_ = "hh";
    }
    ~MOT(){;}

    void matchAndCreateObjects(std::vector<ObjectInView*> &objects_this){
        cv::imshow("ob", objects_this[0]->color_image_);
        cv::waitKey(2);
    }

    void getObjectsStates();

private:
    void hungaryAllocate();

    void getHist();

    void costCalHist();

    void costCalKalman();

    void costCalLabel();



};


#endif //HIST_KALMAN_MOT_MOT_ROS_H
