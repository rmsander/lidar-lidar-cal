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
  if (name == "pc_main.pcd"){
    if (main_done) return;
    main_done = true;
  } else if (name == "pc_front.pcd") {
    if (front_done) return;
    front_done = true;
  } else if (name == "pc_rear.pcd") {
    if (rear_done) return;
    rear_done = true;
  } else {
    std::cerr << "Unrecognized name: " << name << std::endl;
  }

  printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);
  //BOOST_FOREACH (const pcl::PointXYZ& pt, msg->points)
  //  printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
  pcl::io::savePCDFileASCII (name, *msg);
  std::cerr << "Saved " << msg->size() << " data points to file." << std::endl;

}

int main(int argc, char** argv) {
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  //ros::Subscriber sub = nh.subscribe<PointCloud>("/octo_main/octomap_point_cloud_centers", 1, callback);
  std::string name_main = "pc_main.pcd";
  std::string name_front = "pc_front.pcd";
  std::string name_rear = "pc_rear.pcd";
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


