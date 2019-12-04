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
ros::Publisher tracking_objects_pub;
std::map<std::string, int> rviz_objects_max_num;

Eigen::Quaternionf quad(1.0, 0.0, 0.0, 0.0);
double yaw0 = 0.0;
Eigen::Vector3f p0;
double motor_yaw = 0.0;
double motor_yaw_rate = 0.0;
double time_now_to_compare_in_publish = 0.0;
bool if_publish_for_rviz = true;

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

    /** Create  transform matrix**/
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

    /** Find the dynamic objects for tracking **/
    double current_time_stamp = image->header.stamp.toSec();
    std::vector<ObjectInView*> objects_view_this;
    for(const auto & object_i : objects->result)
    {
        if(object_i.label == "person")
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

void publishResultsCallback(const ros::TimerEvent&){

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
    mot.updateCurrentViewField(true, p0, cam_middle_furthest_point_3f, view_field_angle_half_rad, time_now_to_compare_in_publish);

    /** Get all the stored result in tracker and publish. **/
    std::vector<ObjectTrackingResult*> results;
    mot.getObjectsStates(results);

    std::map<std::string, int> rviz_object_counter;
    std::map<std::string, int>::iterator rviz_object_iter;

    static tf::TransformBroadcaster br_ros;
    static tf::Transform transform_ros;

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

        /** Publish tf to show in rviz.  Rviz_objects_max_num is the number of objects in urdf file for rviz **/
        if(if_publish_for_rviz){
            if(rviz_objects_max_num.count(object.label) > 0){
                if(rviz_object_counter.count(object.label) > 0){
                    if(rviz_object_counter[object.label] < rviz_objects_max_num[object.label]-1){
                        rviz_object_counter[object.label] ++;
                    }else{
                        continue;
                    }
                }else{
                    rviz_object_counter[object.label] = 0;
                }

                Eigen::Vector3f predicted_position_now =result_i->position_ + result_i->velocity_ * (ros::Time::now().toSec()-result_i->last_observed_time_);
                transform_ros.setOrigin( tf::Vector3(object.position.x, object.position.y, object.position.z));
                transform_ros.setRotation( tf::Quaternion(0, 0, 0, 1) );
                br_ros.sendTransform(tf::StampedTransform(transform_ros, ros::Time::now(), "world", object.label+std::to_string(rviz_object_counter[object.label])+"_link"));
            }
        }
    }

    tracking_objects_pub.publish(objects_msg);

    /** Let objects that is not updated disappear **/
    if(if_publish_for_rviz){
        for(rviz_object_iter = rviz_objects_max_num.begin(); rviz_object_iter != rviz_objects_max_num.end(); rviz_object_iter++){
            if(rviz_object_counter.count(rviz_object_iter->first) > 0){  //if updated in this callback
                for(int i=rviz_object_counter[rviz_object_iter->first] + 1; i<rviz_object_iter->second; i++){
                    transform_ros.setOrigin( tf::Vector3(1000, 1000, 0));  //fly away
                    transform_ros.setRotation( tf::Quaternion(0, 0, 0, 1) );
                    br_ros.sendTransform(tf::StampedTransform(transform_ros, ros::Time::now(), "world", rviz_object_iter->first+std::to_string(i)+"_link"));
                }
            } else{
                for(int j=0; j<rviz_object_iter->second; j++){
                    transform_ros.setOrigin( tf::Vector3(1000, 1000, 0));  //fly away
                    transform_ros.setRotation( tf::Quaternion(0, 0, 0, 1) );
                    br_ros.sendTransform(tf::StampedTransform(transform_ros, ros::Time::now(), "world", rviz_object_iter->first+std::to_string(j)+"_link"));
                }
            }
        }
    }
}

void positionCallback(const geometry_msgs::PoseStamped& msg)
{
    /** Change from ENU to NWU, NEEDS CAREFUL CHECKING!!!!, chg**/
    time_now_to_compare_in_publish = msg.header.stamp.toSec();

    p0(0) = msg.pose.position.y;
    p0(1) = -msg.pose.position.x;
    p0(2) = msg.pose.position.z;

    quad.x() = msg.pose.orientation.x;
    quad.y() = msg.pose.orientation.y;
    quad.z() = msg.pose.orientation.z;
    quad.w() = msg.pose.orientation.w;

    Eigen::Quaternionf q1(0, 0, 0, 1);
    Eigen::Quaternionf axis = quad * q1 * quad.inverse();
    axis.w() = cos(-PI_2/2.0);
    axis.x() = axis.x() * sin(-PI_2/2.0);
    axis.y() = axis.y() * sin(-PI_2/2.0);
    axis.z() = axis.z() * sin(-PI_2/2.0);
    quad = quad * axis;
    /// Update yaw0 here, should be among [-PI, PI]
    yaw0 = atan2(2*(quad.w()*quad.z()+quad.x()*quad.y()), 1-2*(quad.z()*quad.z()+quad.y()*quad.y()));

    /** For visualization in Rviz.**/
    if(if_publish_for_rviz){
        static tf::TransformBroadcaster br_ros;  // For visualiztion 2 Dec.
        static tf::Transform transform_ros;   // For visualiztion 2 Dec.
        transform_ros.setOrigin( tf::Vector3(p0(0), p0(1), p0(2)));
        transform_ros.setRotation( tf::Quaternion(quad.x(), quad.y(), quad.z(), quad.w()) );
        br_ros.sendTransform(tf::StampedTransform(transform_ros, ros::Time::now(), "world", "uav_link"));
    }
}

void motorCallback(const geometry_msgs::Point32& msg)
{
    static bool init_time = true;
    static double init_head_yaw = 0.0;

    if(init_time)
    {
        init_head_yaw = msg.x;
        init_time = false;
        ROS_INFO("Head Init Yaw in motor coordinate=%f", init_head_yaw);
    }
    else
    {
        motor_yaw = -msg.x + init_head_yaw; // + PI_2?? //start with zero, original z for motor is down. now turn to ENU coordinate. Head forward is PI/2 ???????????
        motor_yaw_rate = -msg.y;

        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX())); //roll
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())); //pitch
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(motor_yaw,Eigen::Vector3d::UnitZ())); //yaw

        Eigen::Quaterniond quaternion;
        quaternion=yawAngle*pitchAngle*rollAngle;

        /** For visualization in Rviz.**/
        if(if_publish_for_rviz){
            static tf::TransformBroadcaster br;
            static tf::Transform transform;
            transform.setOrigin( tf::Vector3(0,0,0));
            transform.setRotation( tf::Quaternion(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()) );
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "uav_link", "head_link"));
        }
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "mot_ros");

    /** Set parameters of MOT for better performance. Optional. **/
    mot.setCandidateOutThreshold(5.0, 6.4);
    mot.setMultiCueCoefficients(0.3f, 0.3f, 0.4f);
    mot.setSimilarityGateValues(0.1f, 0.1f, 0.2f);

    std::map<std::string, float> sigma_acc_map;
    sigma_acc_map["person"] = 1.5f;
    sigma_acc_map["car"] = 2.f;
    mot.setAccelerationVarianceMap(sigma_acc_map);
    mot.setShoudSeeButNotToleration(10);

    /** Other initialization **/
    rviz_objects_max_num["person"] = 3;
    rviz_objects_max_num["ardrone"] = 3;

    p0 << 0.0, 0.0, 0.0;

    /** Define callbacks **/
    ros::NodeHandle nh;
    tracking_objects_pub = nh.advertise<hist_kalman_mot::ObjectsInTracking>("/mot/objects_in_tracking", 1);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/yolo_ros_real_pose/img_for_detected_objects", 1);
    message_filters::Subscriber<yolo_ros_real_pose::ObjectsRealPose> info_sub(nh, "/yolo_ros_real_pose/detected_objects", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, yolo_ros_real_pose::ObjectsRealPose> sync(image_sub, info_sub, 10);
    sync.registerCallback(boost::bind(&objectsCallback, _1, _2));

    ros::Subscriber position_isolate_sub =  nh.subscribe("/mavros/local_position/pose", 1, positionCallback);
    ros::Subscriber motor_sub = nh.subscribe("/place_velocity_info", 1, motorCallback);

    ros::Timer timer = nh.createTimer(ros::Duration(0.1), publishResultsCallback); /// Set a faster timer if the detector is faster.

    ros::spin();
    return 0;
}