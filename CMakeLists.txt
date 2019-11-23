cmake_minimum_required(VERSION 2.8.3)
project(hist_kalman_mot)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++11 -g -O3 -ftree-vectorize -march=native)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_filters
)

find_package(OpenCV 4.1.0 REQUIRED)

catkin_package(
)


include_directories(
# include
        ${catkin_INCLUDE_DIRS}
)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/hist_kalman_mot.cpp
# )


add_executable(mot src/mot.cpp)
target_link_libraries(mot ${OpenCV_LIBS} ${catkin_LIBRARIES})
