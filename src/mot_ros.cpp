//
// Created by cc on 2019/11/23.
//

#include <ros/ros.h>
#include <mot.h>
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <yolo_ros_real_pose/ObjectsRealPose.h>
#include <hist_kalman_mot/ObjectInTracking.h>
#include <hist_kalman_mot/ObjectsInTracking.h>

MOT mot;
ros::Publisher tracking_objects_pub;

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

    /** Find the dynamic objects for tracking **/
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

    /** If any dynamic object is detected, start matching. When there are two people insight.
     * The matching process time is within 5ms and CPU usage is less than 8% on a intel i5-6500 CPU**/
//    double start_time = ros::Time::now().toSec();
    Eigen::Vector3f camera_position;
    camera_position << 0.f, 0.f, 0.f; /// If camera is fixed, use zeros. The position of the camera and the objects should be measured in a same global coordinate
    mot.matchAndCreateObjects(objects_view_this, camera_position);
//    std::cout << "Update time is " << ros::Time::now().toSec() - start_time << std::endl;

    /** Show on image **/
    for(const auto & result_i : objects_view_this)
    {
        cv::rectangle(image_this, result_i->img_pixel_rect_, result_i->color_to_show_, 2);
        cv::putText(image_this, result_i->name_, cv::Point(result_i->img_pixel_rect_.x, result_i->img_pixel_rect_.y),
                cv::FONT_HERSHEY_SIMPLEX,1, result_i->color_to_show_, 2);
    }
    cv::imshow("result", image_this);
    cv::waitKey(1);
//    std::cout << "One callback processed!" << std::endl;

    /** Get all the stored result in tracker and publish. **/
    std::vector<ObjectTrackingResult*> results;
    mot.getObjectsStates(results);

    hist_kalman_mot::ObjectsInTracking objects_msg;
    objects_msg.header.stamp = ros::Time::now();
    for(const auto & result_i : results){
        hist_kalman_mot::ObjectInTracking object;
        object.name = result_i->name_;
        object.label = result_i->label_;
        object.position.x = result_i->position_[0];
        object.position.y = result_i->position_[1];
        object.position.z = result_i->position_[2];
        object.velocity.x = result_i->velocity_[0];
        object.velocity.y = result_i->velocity_[1];
        object.velocity.z = result_i->velocity_[2];
        object.last_observed_time = result_i->last_observed_time_;
        object.sigma = result_i->sigma_;
        objects_msg.result.push_back(object);
    }
    tracking_objects_pub.publish(objects_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mot_ros");

    /** Set parameters of MOT for better performance. Optional. **/
    mot.setCandidateOutThreshold(10.0, 6.4);
    mot.setMultiCueCoefficients(0.3f, 0.3f, 0.4f);
    mot.setSimilarityGateValues(0.001f, 0.1f, 0.2f);

    std::map<std::string, float> sigma_acc_map;
    sigma_acc_map["person"] = 1.5f;
    sigma_acc_map["car"] = 2.f;
    mot.setAccelerationVarianceMap(sigma_acc_map);

    /** Define callbacks **/
    ros::NodeHandle nh;
    tracking_objects_pub = nh.advertise<hist_kalman_mot::ObjectsInTracking>("/mot/objects_in_tracking", 1);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/yolo_ros_real_pose/img_for_detected_objects", 1);
    message_filters::Subscriber<yolo_ros_real_pose::ObjectsRealPose> info_sub(nh, "/yolo_ros_real_pose/detected_objects", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, yolo_ros_real_pose::ObjectsRealPose> sync(image_sub, info_sub, 10);
    sync.registerCallback(boost::bind(&objectsCallback, _1, _2));

    ros::spin();
    return 0;
}