//
// Created by cc on 2019/11/23.
//

#include <ros/ros.h>
#include <mot.h>
#include <object_in_view.h>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <yolo_ros_real_pose/ObjectsRealPose.h>

using namespace std;

MOT mot;
vector<ObjectInView*> detected_objects;

void objectsCallback(const sensor_msgs::ImageConstPtr& image, const yolo_ros_real_pose::ObjectsRealPoseConstPtr& objects)
{
    if(objects->result.empty()){
        //ROS_INFO("No objects in view!");
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvShare(image); //sensor_msgs::image_encodings::RGB8
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat image_this = cv_ptr->image;

    std::vector<ObjectInView*> objects_view_this;

    for(const auto & object_i : objects->result)
    {
        if(object_i.label == "person")
        {
            auto* ob_temp = new ObjectInView();
            ob_temp->name_ = object_i.label;
            ob_temp->observed_time_ = image->header.stamp.toSec();
            ob_temp->color_image_ = image_this(cv::Range(object_i.pix_lt_y, object_i.pix_rb_y),
                                               cv::Range(object_i.pix_lt_x, object_i.pix_rb_x));
            ob_temp->label_ = object_i.label;
            ob_temp->label_confidence_ = object_i.confidence;
            /// Set detected real position
            ob_temp->position_[0] = object_i.x;
            ob_temp->position_[1] = object_i.y;
            ob_temp->position_[2] = object_i.z;

            objects_view_this.push_back(ob_temp);
        }
    }

    if(objects_view_this.empty()){
        //ROS_INFO("Found objects. But no dynmaic objects!");
        return;
    }

    mot.matchAndCreateObjects(objects_view_this);
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "mot_ros");
    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/yolo_ros_real_pose/img_for_detected_objects", 1);
    message_filters::Subscriber<yolo_ros_real_pose::ObjectsRealPose> info_sub(nh, "/yolo_ros_real_pose/detected_objects", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, yolo_ros_real_pose::ObjectsRealPose> sync(image_sub, info_sub, 10);
    sync.registerCallback(boost::bind(&objectsCallback, _1, _2));

    ros::spin();
    return 0;
}