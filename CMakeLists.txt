cmake_minimum_required(VERSION 2.8.3)
project(hist_kalman_mot)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++11 -g -O2 -ftree-vectorize -march=native)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    message_filters
    cv_bridge
    sensor_msgs
    image_transport
    yolo_ros_real_pose
)

find_package(OpenCV REQUIRED)

catkin_package(
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(${EIGEN_DEFINITIONS})

include_directories(
    ${catkin_INCLUDE_DIRS}
)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/hist_kalman_mot.cpp
# )

message(${OpenCV_LIBRARIES})

add_executable(mot src/mot_ros.cpp)
target_link_libraries(mot ${OpenCV_LIBS} ${catkin_LIBRARIES})
