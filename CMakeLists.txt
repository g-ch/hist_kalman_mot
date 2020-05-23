cmake_minimum_required(VERSION 2.8.3)
project(hist_kalman_mot)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++11 -g -O1 -ftree-vectorize -march=native)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    message_filters
    cv_bridge
    sensor_msgs
    image_transport
    message_generation
    tf
)

find_package(OpenCV REQUIRED)

add_message_files(
    FILES
    ObjectInTracking.msg
    ObjectsInTracking.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("../../devel/include")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(${EIGEN_DEFINITIONS})

#message(${catkin_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/usr/local/include/munkres)
set(MunkresLIB
        /usr/local/lib/libmunkres.a
        )

add_executable(mot src/mot_ros.cpp)
add_dependencies(mot        ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
target_link_libraries(mot ${OpenCV_LIBS} ${catkin_LIBRARIES} ${MunkresLIB})
