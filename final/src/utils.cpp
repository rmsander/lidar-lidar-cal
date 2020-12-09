#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <fstream>


int loadCloud(std::string name, pcl::PointCloud<pcl::PointXYZ>::Ptr ptr){

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (name, *ptr) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  std::cout << "Loaded "
            << ptr->width * ptr->height
            << " data points from " << name
            << std::endl;
  return 0;
}


Eigen::Matrix4d readMatrix4(const char *filename){

    double buff[4][4];

    std::ifstream infile;
    infile.open(filename);

    for (auto ix = 0; ix < 4; ++ix) {
        std::string line;
        std::getline(infile, line);
        std::stringstream stream(line);
        for (auto jx = 0; jx < 4; ++jx) {
            stream >> buff[ix][jx];
        }
    }

    Eigen::Matrix4d T;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            T(i,j) = buff[i][j];


    return T;
}


void filterAndPrint(std::string file_name, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, double leaf_size) {

  // Create the filtering object
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud_in);
  sor.setLeafSize (leaf_size, leaf_size, leaf_size);
  sor.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
       << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;
    pcl::io::savePCDFileASCII (file_name, *cloud_filtered);

    std::cerr << "Saved " << cloud_filtered->size() << " data points to file. (Down from " << cloud_in->size() << " points)" << std::endl;
}

