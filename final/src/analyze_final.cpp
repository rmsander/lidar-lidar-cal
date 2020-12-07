#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <stdlib.h>

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

int loadCloud(std::string name, pcl::PointCloud<pcl::PointXYZ>::Ptr ptr){

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (name, *ptr) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  //std::cout << "Loaded "
  //          << ptr->width * ptr->height
  //          << " data points from " << name
  //          << std::endl;
  return 0;
}

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
            
        //} else if (argc > 3) {
        //    double voxel_size = std::stof(argv[2]);
        //    int n_runs = atoi(argv[3]);
        //    for (auto ix = 1; ix <= n_runs; ++ix) {

        //        char main_in_buff[100];
        //        snprintf(main_in_buff, sizeof(main_in_buff), "main_clouds/pc_%05d_vox_%.2f.pcd", 30*ix, voxel_size);
        //        std::string main_in = main_in_buff;

        //        char front_in_buff[100];
        //        snprintf(front_in_buff, sizeof(front_in_buff), "front_clouds/pc_%05d_vox_%.2f.pcd", 30*ix, voxel_size);
        //        std::string front_in = front_in_buff;

        //        // Main cloud
        //        pcl::PointCloud<pcl::PointXYZ>::Ptr main_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        //        loadCloud(main_in, main_cloud);

        //        // Front cloud
        //        pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        //        loadCloud(front_in, front_cloud);

        //        pcl::registration::TransformationValidationEuclidean<pcl::PointXYZ, pcl::PointXYZ> tve;
        //        tve.setMaxRange (maxrange);
        //        Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
        //        double score = tve.validateTransformation (main_cloud, front_cloud, identity);
        //        std::cout << 30*ix << " scans. Score: " << score << std::endl;

        //    }
        } else {
            //TODO : Make this runnable
            // Main cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr main_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            //std::string main_name = "main_out.pcd";
            std::string main_name = "main_clouds/main_cloud_full.pcd";
            loadCloud(main_name, main_cloud);

            // Front cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            //std::string front_name = "front_out.pcd";
            std::string front_name = "front_clouds/front_out_opt2_calibration.pcd";
            loadCloud(front_name, front_cloud);

            pcl::registration::TransformationValidationEuclidean<pcl::PointXYZ, pcl::PointXYZ> tve;
            tve.setMaxRange (maxrange);
            Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
            double score = tve.validateTransformation (main_cloud, front_cloud, identity);
            std::cout << score << std::endl;
        }

  return (0);
}


