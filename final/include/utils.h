#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <fstream>


int loadCloud(std::string name, pcl::PointCloud<pcl::PointXYZ>::Ptr ptr);

Eigen::Matrix4d readMatrix4(const char *filename);

void filterAndPrint(std::string file_name, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, double leaf_size);
