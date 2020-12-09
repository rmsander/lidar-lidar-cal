#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <stdlib.h>
#include "utils.h"

int main (int argc, char** argv) {

        double maxrange = 0.5;
        if (argc > 1){
            maxrange = std::stof(argv[1]);
        }

        if (argc == 2) {
            std::cout << "Using maxrange: " << maxrange << std::endl;
            //todo: make this settable from commandline
            int n_scales = 8;
            int n_samples = 10;
            std::string main_in = "main_clouds/main_cloud_full.pcd";
            // Main cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr main_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            loadCloud(main_in, main_cloud);
            for (int scale_ix = 0; scale_ix < n_scales; ++scale_ix) {
                std::vector<double> score_vec;
                for (int sample_ix=0; sample_ix < n_samples; ++sample_ix) {
                    std::cout << "Scale, sample: " << scale_ix << "," << sample_ix << std::endl;
                    char front_in_buff[100];
                    snprintf(front_in_buff, sizeof(front_in_buff), "front_clouds/pc_scaleix_%d_sampleix_%d.pcd", scale_ix, sample_ix);
                    std::string front_in = front_in_buff;

                    // Front cloud
                    pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud (new pcl::PointCloud<pcl::PointXYZ>);
                    loadCloud(front_in, front_cloud);

                    pcl::registration::TransformationValidationEuclidean<pcl::PointXYZ, pcl::PointXYZ> tve;
                    tve.setMaxRange (maxrange);
                    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
                    double score = tve.validateTransformation (main_cloud, front_cloud, identity);
                    score_vec.push_back(score);
                }

                char fname[100];
                snprintf(fname, sizeof(fname), "front_clouds/scaleix_%d_results.txt", scale_ix);
                std::string fname_str = fname;

                std::ofstream myfile;
                myfile.open (fname_str);
                for (double score : score_vec) {
                    myfile << score << std::endl;
                }
                myfile.close();
            }
            std::cout << "Finished!" << std::endl;
            
        }

  return (0);
}


