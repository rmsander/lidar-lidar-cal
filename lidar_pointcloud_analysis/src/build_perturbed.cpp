#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"

Eigen::Matrix4d rpyToHomog(double r, double p, double y) {
//Careful, not guarantees on order

    Eigen::Matrix4d R;
    R << cos(r), sin(r), 0, 0,
         -sin(r), cos(r), 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;

    Eigen::Matrix4d P;
    P << 1, 0, 0, 0,
         0, cos(p), sin(p), 0,
         0, -sin(p), cos(p), 0,
         0, 0, 0, 1;

    Eigen::Matrix4d Y;
    Y << cos(y), 0, -sin(y),  0,
         0, 1, 0, 0,
         sin(y), 0, cos(y), 0,
         0, 0, 0, 1;

    return R * P * Y;

}


int main (int argc, char** argv) {
    // sample_offset n_samples scale_offset scale1 scale2 ... scalen

    int sample_offset = atoi(argv[1]);
    int n_rot_perturb_each = atoi(argv[2]);
    int scale_offset = atoi(argv[3]);
    std::vector<double> rot_perturb_scales;
    for (auto ix = 4; ix < argc; ++ix) {
        rot_perturb_scales.push_back(std::stof(argv[ix]));
    }

    int start_scan = 0;
    int end_scan = 150;

    double voxel_size = 0.05;

    Eigen::Matrix4d T_front_main = readMatrix4("velodyne_front.txt");

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> main_clouds;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> front_clouds;
    std::vector<Eigen::Matrix4d> transforms;
    for (auto ix = start_scan; ix < end_scan; ++ix) {
            char buff1[100];
            snprintf(buff1, sizeof(buff1), "main_lidar/pc_%05d.pcd", ix);
            std::string pc_name = buff1;

            char buff_front[100];
            snprintf(buff_front, sizeof(buff_front), "front_lidar/pc_%05d.pcd", ix);
            std::string pc_name_front = buff_front;

            char buff_pose[100];
            snprintf(buff_pose, sizeof(buff_pose), "main_lidar_pose/pose_%05d.txt", ix);
            std::string pose_name = buff_pose;
            Eigen::Matrix4d T = readMatrix4(pose_name.c_str());

            // Main cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            loadCloud(pc_name, temp_cloud);

            // Front Cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
            loadCloud(pc_name_front, temp_cloud_front);

            main_clouds.push_back(temp_cloud);
            front_clouds.push_back(temp_cloud_front);
            transforms.push_back(T);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud_main (new pcl::PointCloud<pcl::PointXYZ>);
    for (auto ix = start_scan; ix < end_scan; ++ix) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        temp_cloud = main_clouds[ix];
        Eigen::Matrix4d T = transforms[ix];
        
        // Executing the transformation
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::transformPointCloud (*temp_cloud, *transformed_cloud, T);


        (*full_cloud_main) += (*transformed_cloud);
        std::cout << full_cloud_main->size() << std::endl;
    }

    char main_out_buff[100];
    snprintf(main_out_buff, sizeof(main_out_buff), "main_clouds/main_cloud_full.pcd");
    std::string main_out = main_out_buff;
    filterAndPrint(main_out, full_cloud_main, voxel_size);

    for (auto scale_index = 0; scale_index < rot_perturb_scales.size(); ++scale_index){
        for (auto sample_index = 0; sample_index < n_rot_perturb_each; ++sample_index) {
            std::cout << "Scale ix, sample ix: " << scale_index << "," << sample_index << std::endl;
            double r = rot_perturb_scales[scale_index] * static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
            double p = rot_perturb_scales[scale_index] * static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
            double y = rot_perturb_scales[scale_index] * static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

            Eigen::Matrix4d perturbation_matrix = rpyToHomog(r, p, y);

            pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
            for (auto ix = start_scan; ix < end_scan; ++ix) {

                pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
                temp_cloud_front = front_clouds[ix];
                Eigen::Matrix4d T = transforms[ix];
                
                // Executing the transformation
                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_front (new pcl::PointCloud<pcl::PointXYZ> ());
                Eigen::Matrix4d front_main_trans = T * perturbation_matrix * T_front_main;
                pcl::transformPointCloud (*temp_cloud_front, *transformed_cloud_front, front_main_trans);

                (*full_cloud_front) += (*transformed_cloud_front);

            }

            char front_out_buff[100];
            snprintf(front_out_buff, sizeof(front_out_buff), "front_clouds/pc_scaleix_%d_sampleix_%d.pcd", scale_offset + scale_index, sample_offset + sample_index);
            std::string front_out = front_out_buff;
            filterAndPrint(front_out, full_cloud_front, voxel_size);
        }
    }

  return (0);
}


