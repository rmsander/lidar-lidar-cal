#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <iostream>
#include <functional>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

bool main_done = false;
bool front_done = false;
bool rear_done = false;

void callback(std::string name, const PointCloud::ConstPtr& msg) {
  if (name.find("pc_main.pcd") != std::string::npos){
    if (main_done) return;
    main_done = true;
  } else if (name.find("pc_front.pcd") != std::string::npos){
    if (front_done) return;
    front_done = true;
  } else if (name.find("pc_rear.pcd") != std::string::npos){
    if (rear_done) return;
    rear_done = true;
  } else {
    std::cerr << "Unrecognized name: " << name << std::endl;
  }

  printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);
  pcl::io::savePCDFileASCII (name, *msg);
  std::cerr << "Saved " << msg->size() << " data points to file." << std::endl;

}

int main(int argc, char** argv) {
  std::string prefix;
  if (argc > 1){
    prefix = argv[1];
  } else {
    prefix = "";
  }
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  std::string name_main = prefix + "pc_main.pcd";
  std::string name_front = prefix + "pc_front.pcd";
  std::string name_rear = prefix + "pc_rear.pcd";
  ros::Subscriber sub_main = nh.subscribe<PointCloud>("/octo_main/octomap_point_cloud_centers", 1, [name_main](const PointCloud::ConstPtr& msg){callback(name_main, msg);});
  ros::Subscriber sub_front = nh.subscribe<PointCloud>("/octo_front/octomap_point_cloud_centers", 1, [name_front](const PointCloud::ConstPtr& msg){callback(name_front, msg);});
  ros::Subscriber sub_rear = nh.subscribe<PointCloud>("/octo_rear/octomap_point_cloud_centers", 1, [name_rear](const PointCloud::ConstPtr& msg){callback(name_rear, msg);});

  while (ros::ok()) {
      ros::spinOnce();
      if (main_done && front_done && rear_done) {
          break;
      }
  }
}


