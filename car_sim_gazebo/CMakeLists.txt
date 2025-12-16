cmake_minimum_required(VERSION 3.5)
project(car_sim_gazebo)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(rclcpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(visualization_msgs REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# bundled quadprog (header-only interface + .cc implementation)
set(QUADPROG_SOURCES
  mpc_lateral/Array.cc
  mpc_lateral/QuadProg++.cc
)
add_library(quadprog STATIC ${QUADPROG_SOURCES})
target_include_directories(quadprog PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/quadprog
  ${PROJECT_SOURCE_DIR}/mpc_lateral
)

add_library(jarbay_plugin SHARED
  src/jarbay_plugin.cpp
)
ament_target_dependencies(jarbay_plugin
  rclcpp
  gazebo_ros
  tf2_ros
  nav_msgs
  geometry_msgs
  ackermann_msgs
)
target_include_directories(jarbay_plugin PUBLIC
  include
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(jarbay_plugin
  ${GAZEBO_LIBRARIES}
)

add_executable(mpc_lateral_node
  mpc_lateral/mpc_lateral.cpp
)
ament_target_dependencies(mpc_lateral_node
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_ros
  tf2
  tf2_geometry_msgs
  visualization_msgs
)
target_include_directories(mpc_lateral_node PUBLIC
  include
  ${PROJECT_SOURCE_DIR}/include/quadprog
  ${PROJECT_SOURCE_DIR}/mpc_lateral
  ${EIGEN3_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpc_lateral_node
  quadprog
  Eigen3::Eigen
)

### Install Rules
install(
  TARGETS
    jarbay_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    mpc_lateral_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
    worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
  scripts/teleop_keyboard.py
  scripts/odom_tf_broadcaster.py
  route_finding/path_recorder.py
  route_finding/csv_path_player.py
  purepursuit/pure_pursuit.py
  trajectory_viewer/trajectory.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
