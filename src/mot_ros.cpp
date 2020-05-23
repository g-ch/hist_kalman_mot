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
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point32.h>

#define PIx2 6.28318
#define PI 3.14159
#define PI_2 1.5708

MOT mot;
ros::Publisher tracking_objects_pub, tracking_prediction_pub;

Eigen::Quaternionf quad(1.0, 0.0, 0.0, 0.0);
double yaw0 = 0.0;
Eigen::Vector3f p0;
double motor_yaw = 0.0;
double motor_yaw_rate = 0.0;
double time_now_to_compare_in_publish = 0.0;

bool if_in_simulation = false;


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
    /** Update uav pose and motor yaw in this frame **/

    if(!if_in_simulation){
        p0(0) = objects->result[0].local_pose.position.y;
        p0(1) = -objects->result[0].local_pose.position.x;
        p0(2) = objects->result[0].local_pose.position.z;

        quad.x() = objects->result[0].local_pose.orientation.x;
        quad.y() = objects->result[0].local_pose.orientation.y;
        quad.z() = objects->result[0].local_pose.orientation.z;
        quad.w() = objects->result[0].local_pose.orientation.w;

        Eigen::Quaternionf q1(0, 0, 0, 1);
        Eigen::Quaternionf axis = quad * q1 * quad.inverse();
        axis.w() = cos(-PI_2/2.0);
        axis.x() = axis.x() * sin(-PI_2/2.0);
        axis.y() = axis.y() * sin(-PI_2/2.0);
        axis.z() = axis.z() * sin(-PI_2/2.0);
        quad = quad * axis;
        /// Update yaw0 here, should be among [-PI, PI] 
        yaw0 = atan2(2*(quad.w()*quad.z()+quad.x()*quad.y()), 1-2*(quad.z()*quad.z()+quad.y()*quad.y()));

	    std::cout << "Mot yaw0="<<yaw0<<std::endl;
    }else{
        p0(0) = objects->result[0].local_pose.position.x;
        p0(1) = objects->result[0].local_pose.position.y;
        p0(2) = objects->result[0].local_pose.position.z;

        quad.x() = objects->result[0].local_pose.orientation.x;
        quad.y() = objects->result[0].local_pose.orientation.y;
        quad.z() = objects->result[0].local_pose.orientation.z;
        quad.w() = objects->result[0].local_pose.orientation.w;
        yaw0 = atan2(2*(quad.w()*quad.z()+quad.x()*quad.y()), 1-2*(quad.z()*quad.z()+quad.y()*quad.y()));
    }

    

    // static bool init_head_time = true;
    // static double init_head_yaw = 0.0;
    // if(init_head_time){
    //     init_head_yaw = objects->result[0].head_yaw;
    //     init_head_time = false;
    //     ROS_INFO("Head Init Yaw in motor coordinate=%f", init_head_yaw);
    // }
    // else {
    //     motor_yaw = -objects->result[0].head_yaw + init_head_yaw;
    // }
    motor_yaw = objects->result[0].head_yaw;


    /** Create  transform matrix**/
    Eigen::Quaternionf axis_motor;
    axis_motor.w() = cos(motor_yaw/2.0);
    axis_motor.x() = 0;
    axis_motor.y() = 0;
    axis_motor.z() = sin(motor_yaw/2.0);
    Eigen::Quaternionf quad_rotate = quad * axis_motor;

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = Eigen::Matrix3f(quad_rotate);
    transform(0, 3) = p0(0);
    transform(1, 3) = p0(1);
    transform(2, 3) = p0(2);

    Eigen::Matrix4f t_c_b = Eigen::Matrix4f::Zero();
    t_c_b(0, 2) = 1;
    t_c_b(1, 0) = -1;
    t_c_b(2, 1) = -1;
    t_c_b(3, 3) = 1;

    /** Find the dynamic objects for tracking **/
    double current_time_stamp = image->header.stamp.toSec();
    std::vector<ObjectInView*> objects_view_this;
    for(const auto & object_i : objects->result)
    {
        if(object_i.label == "robot")  //object_i.label == "drone" || 
        {
            auto* ob_temp = new ObjectInView();
            ob_temp->name_ = object_i.label;
            ob_temp->observed_time_ = current_time_stamp;
            ob_temp->color_image_ = image_this(cv::Range(object_i.pix_lt_y, object_i.pix_rb_y),
                                               cv::Range(object_i.pix_lt_x, object_i.pix_rb_x));
            ob_temp->img_pixel_rect_ = cv::Rect(object_i.pix_lt_x, object_i.pix_lt_y, object_i.pix_rb_x-object_i.pix_lt_x, object_i.pix_rb_y-object_i.pix_lt_y);
            ob_temp->label_ = object_i.label;
            ob_temp->label_confidence_ = object_i.confidence;
            /** Set detected real position. If you want to use pixel position for the Kalman filter,
             * just set position position_[i] as pixel position. You may set position_[i]=0 to make it uselsess**/

            Eigen::Vector4f pose_ori, pose_global;
            pose_ori << object_i.x, object_i.y, object_i.z, 1.f;
            pose_global = t_c_b * pose_ori;
            pose_global = transform * pose_global;

            ob_temp->position_[0] = pose_global[0];
            ob_temp->position_[1] = pose_global[1];
            ob_temp->position_[2] = pose_global[2];

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

       mot.matchAndCreateObjects(objects_view_this);

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
}

void publishResultsCallback(const ros::TimerEvent& e){

    /** Create  transform matrix for this time**/
    Eigen::Quaternionf q1(0, 0, 0, 1);
    Eigen::Quaternionf axis = quad * q1 * quad.inverse();
    axis.w() = cos(motor_yaw/2.0);
    axis.x() = axis.x() * sin(motor_yaw/2.0);
    axis.y() = axis.y() * sin(motor_yaw/2.0);
    axis.z() = axis.z() * sin(motor_yaw/2.0);
    Eigen::Quaternionf quad_rotate = quad * axis;

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = Eigen::Matrix3f(quad_rotate);
    transform(0, 3) = p0(0);
    transform(1, 3) = p0(1);
    transform(2, 3) = p0(2);

    Eigen::Matrix4f t_c_b = Eigen::Matrix4f::Zero();
    t_c_b(0, 2) = 1;
    t_c_b(1, 0) = -1;
    t_c_b(2, 1) = -1;
    t_c_b(3, 3) = 1;

    Eigen::Vector4f cam_middle_furthest_point;
    cam_middle_furthest_point << 0, 0, 5.f, 1.f;
    cam_middle_furthest_point = t_c_b * cam_middle_furthest_point;
    cam_middle_furthest_point = transform * cam_middle_furthest_point;
    Eigen::Vector3f cam_middle_furthest_point_3f;
    cam_middle_furthest_point_3f << cam_middle_furthest_point[0], cam_middle_furthest_point[1], cam_middle_furthest_point[2];

    float view_field_angle_half_rad = PI / 8.f; // A little smaller because objects near the edge of the image are usually hard to detect.
    /// If you want delete the objects that should be in the view field but not, use this true.
    time_now_to_compare_in_publish = ros::Time::now().toSec();
    mot.updateCurrentViewField(true, p0, cam_middle_furthest_point_3f, view_field_angle_half_rad, time_now_to_compare_in_publish);

    /** Get all the stored result in tracker and publish. **/
    std::vector<ObjectTrackingResult*> results;
    mot.getObjectsStates(results);


    static tf::TransformBroadcaster br_ros;
    static tf::Transform transform_ros;

    hist_kalman_mot::ObjectsInTracking objects_msg, objects_predicted_msg;
    objects_predicted_msg.header.stamp = objects_msg.header.stamp = ros::Time::now();
    double current_time =  ros::Time::now().toSec();

    for(const auto & result_i : results){
        hist_kalman_mot::ObjectInTracking object;
        object.name = result_i->name_;
        object.label = result_i->label_;
        object.position.x = result_i->position_[0];
        object.position.y = result_i->position_[1];
        object.position.z = 1.2; //result_i->position_[2];  for Obstacles on the ground !!!!!
        object.velocity.x = result_i->velocity_[0];
        object.velocity.y = result_i->velocity_[1];
        object.velocity.z = 0.0; //->velocity_[2];
        object.last_observed_time = result_i->last_observed_time_;
        object.sigma = result_i->sigma_;
        objects_msg.result.push_back(object);

        //predicted position
        double delt_t = current_time - object.last_observed_time;
        object.position.x = result_i->position_[0] + object.velocity.x * delt_t;
        object.position.y = result_i->position_[1] + object.velocity.y * delt_t;;
        object.position.z = result_i->position_[2] + object.velocity.z * delt_t;;
        objects_predicted_msg.result.push_back(object);  
    }

    tracking_objects_pub.publish(objects_msg);
    tracking_prediction_pub.publish(objects_predicted_msg);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "mot_ros");

    /** Set parameters of MOT for better performance. Optional. **/
    mot.setCandidateOutThreshold(5.0, 6.4);  // Time(second), distance(m)
    mot.setMultiCueCoefficients(0.7f, 0.2f, 0.1f); // lamda_position, lamda_hist, lamda_label
    mot.setSimilarityGateValues(0.06f, 0.05f, 0.1f);

    std::map<std::string, float> sigma_acc_map;
    sigma_acc_map["person"] = 1.5f;
    sigma_acc_map["robot"] = 1.5f;
    sigma_acc_map["car"] = 2.f;
    mot.setAccelerationVarianceMap(sigma_acc_map);
    mot.setShoudSeeButNotToleration(10);

    p0 << 0.0, 0.0, 0.0;

    /** Define callbacks **/
    ros::NodeHandle nh;
    tracking_objects_pub = nh.advertise<hist_kalman_mot::ObjectsInTracking>("/mot/objects_in_tracking", 1);
    tracking_prediction_pub = nh.advertise<hist_kalman_mot::ObjectsInTracking>("/mot/objects_in_tracking_predicted", 1);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/yolo_ros_real_pose/img_for_detected_objects", 1);
    message_filters::Subscriber<yolo_ros_real_pose::ObjectsRealPose> info_sub(nh, "/yolo_ros_real_pose/detected_objects", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, yolo_ros_real_pose::ObjectsRealPose> sync(image_sub, info_sub, 10);
    sync.registerCallback(boost::bind(&objectsCallback, _1, _2));

    ros::Timer timer = nh.createTimer(ros::Duration(0.1), publishResultsCallback); /// Set a faster timer if the detector is faster.

    ros::spin();
    return 0;
}
