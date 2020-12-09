#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <iostream>
#include <functional>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

PointCloud::ConstPtr pc_main;
PointCloud::ConstPtr pc_front;
PointCloud::ConstPtr pc_rear;
int pc_index = 0;

bool pc_main_ready = false;
bool pc_front_ready = false;
bool pc_rear_ready = false;

tf2_ros::Buffer tfBuffer;

void pc_callback(std::string name, const PointCloud::ConstPtr& msg) {
  
  printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);
  if (name == "main") {
    pc_main = msg;
    pc_main_ready = true;
  } else if (name == "front") {
    pc_front = msg;
    pc_front_ready = true;
  } else if (name == "rear"){
    pc_rear = msg;
    pc_rear_ready = true;
    }
}

void pcd_save_callback(const ros::TimerEvent& event) {


  if (!(pc_main_ready && pc_front_ready && pc_rear_ready)){
    return;
  }

  Eigen::Affine3d transform_matrix;
  try {
    geometry_msgs::TransformStamped transform_to_robot = tfBuffer.lookupTransform("l1_map", "husky4/velodyne", ros::Time::now(), ros::Duration(1));
    transform_matrix = tf2::transformToEigen(transform_to_robot);
    std::cout << transform_matrix.matrix() << std::endl;

  } catch (tf2::TransformException &ex){
    ROS_WARN("%s",ex.what());
    return;
  }


  char buff1[100];
  snprintf(buff1, sizeof(buff1), "main_lidar/pc_%05d.pcd", pc_index);
  std::string main_pc_name = buff1;
  char buff2[100];
  snprintf(buff2, sizeof(buff2), "rear_lidar/pc_%05d.pcd", pc_index);
  std::string rear_pc_name = buff2;
  char buff3[100];
  snprintf(buff3, sizeof(buff3), "front_lidar/pc_%05d.pcd", pc_index);
  std::string front_pc_name = buff3;
  char buff4[100];
  snprintf(buff4, sizeof(buff4), "main_lidar_pose/pose_%05d.txt", pc_index);
  std::string pose_name = buff4;

  pcl::io::savePCDFileASCII (main_pc_name, *pc_main);
  std::cerr << "Saved " << pc_main->size() << " data points to file." << std::endl;
  pcl::io::savePCDFileASCII (rear_pc_name, *pc_rear);
  std::cerr << "Saved " << pc_rear->size() << " data points to file." << std::endl;
  pcl::io::savePCDFileASCII (front_pc_name, *pc_front);
  std::cerr << "Saved " << pc_front->size() << " data points to file." << std::endl << std::endl;

  std::ofstream file(pose_name);
  if (file.is_open())
  {
    file << transform_matrix.matrix() << std::endl;
  }

  pc_index += 1;
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "ros_to_pcl");
    ros::NodeHandle nh;
    tf2_ros::TransformListener tfListener(tfBuffer);


    std::string main_name = "main";
    std::string front_name = "front";
    std::string rear_name = "rear";
    ros::Subscriber sub_main = nh.subscribe<PointCloud>("/husky4/velodyne_points", 1, [main_name](const PointCloud::ConstPtr& msg){pc_callback(main_name, msg);});
    ros::Subscriber sub_front = nh.subscribe<PointCloud>("/husky4/velodyne_front/velodyne_points", 1, [front_name](const PointCloud::ConstPtr& msg){pc_callback(front_name, msg);});
    ros::Subscriber sub_rear = nh.subscribe<PointCloud>("/husky4/velodyne_rear/velodyne_points", 1, [rear_name](const PointCloud::ConstPtr& msg){pc_callback(rear_name, msg);});

    ros::Timer save_timer = nh.createTimer(ros::Duration(1./2.), pcd_save_callback);

    ros::spin();
}


