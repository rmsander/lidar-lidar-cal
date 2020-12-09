#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <stdlib.h>
#include "utils.h"

int main (int argc, char** argv) {

        if (argc != 4) {
            std::cout << "Wrong number of arguments! Form is: ./analyze_single maxrange filename_1 filename_2" << std::endl;
            return 1;
        }
        double maxrange = std::stof(argv[1]);
        std::string fn1 = argv[2];
        std::string fn2 = argv[3];

        // Main cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr main_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        //std::string main_name = "main_out.pcd";
        //std::string main_name = "main_clouds/main_cloud_full.pcd";
        loadCloud(fn1, main_cloud);

        // Front cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        //std::string front_name = "front_out.pcd";
        //std::string front_name = "front_clouds/front_out_opt2_calibration.pcd";
        loadCloud(fn2, front_cloud);

        pcl::registration::TransformationValidationEuclidean<pcl::PointXYZ, pcl::PointXYZ> tve;
        tve.setMaxRange (maxrange);
        Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
        double score = tve.validateTransformation (main_cloud, front_cloud, identity);
        std::cout << score << std::endl;

  return (0);
}


