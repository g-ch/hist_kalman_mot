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
    static double last_update_time = 0.0;
    if(image->header.stamp.toSec() < last_update_time){
        std::cout << "$$$$$$$$$$ Time stamp in callback makes no sense! $$$$$$$$" << std::endl;
        return;
    }
    last_update_time = image->header.stamp.toSec();

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
            ob_temp->img_pixel_rect_ = cv::Rect(object_i.pix_lt_x, object_i.pix_lt_y, object_i.pix_rb_x-object_i.pix_lt_x, object_i.pix_rb_y-object_i.pix_lt_y);
            ob_temp->label_ = object_i.label;
            ob_temp->label_confidence_ = object_i.confidence;
            /** Set detected real position. If you want to use pixel position for the Kalman filter,
             * just set position position_[i] as pixel position. You may set position_[i]=0 to make it uselsess**/
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

    /** If any dynamic object is detected, start matching. **/
    mot.matchAndCreateObjects(objects_view_this);

    /** Show on image **/
    for(const auto & result_i : objects_view_this)
    {
        cv::rectangle(image_this, result_i->img_pixel_rect_, result_i->color_to_show_, 2);
        cv::putText(image_this, result_i->name_, cv::Point(result_i->img_pixel_rect_.x, result_i->img_pixel_rect_.y),
                cv::FONT_HERSHEY_SIMPLEX,1, result_i->color_to_show_, 2);
    }
    cv::imshow("result", image_this);
    cv::waitKey(1);
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