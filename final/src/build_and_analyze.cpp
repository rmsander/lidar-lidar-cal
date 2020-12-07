#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <stdlib.h>

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
  std::cout << "Loaded "
            << ptr->width * ptr->height
            << " data points from " << name
            << std::endl;
  return 0;
}

int main (int argc, char** argv) {

    int start_scan = 0;
    int end_scan = 50;
    int n_scans = 589;
    if (argc > 2) {
        start_scan = atoi(argv[1]);
        end_scan = atoi(argv[2]);
    }
    double voxel_size = 0.03;
    if (argc > 3) {
        voxel_size = std::stof(argv[3]);
    }
        

    Eigen::Matrix4d T_front_main = readMatrix4("velodyne_front_optimized2.txt");
    //Eigen::Matrix4d T_front_main = readMatrix4("velodyne_front_optimized.txt");
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud_main (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
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

         // Executing the transformation
         pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
         // You can either apply transform_1 or transform_2; they are the same
         pcl::transformPointCloud (*temp_cloud, *transformed_cloud, T);


        (*full_cloud_main) += (*transformed_cloud);
        std::cout << full_cloud_main->size() << std::endl;

        // Front Cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
        loadCloud(pc_name_front, temp_cloud_front);

         // Executing the transformation
         pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_front (new pcl::PointCloud<pcl::PointXYZ> ());
        Eigen::Matrix4d front_main_trans = T * T_front_main;
         pcl::transformPointCloud (*temp_cloud_front, *transformed_cloud_front, front_main_trans);

        (*full_cloud_front) += (*transformed_cloud_front);

        //was used to generate sweep
        //if ( (ix > 0) && (ix % 30 == 0)) {
        //    char main_out_buff[100];
        //    snprintf(main_out_buff, sizeof(main_out_buff), "main_clouds/pc_%05d_vox_%.2f.pcd", ix, voxel_size);
        //    std::string main_out = main_out_buff;
        //    char front_out_buff[100];
        //    snprintf(front_out_buff, sizeof(front_out_buff), "front_clouds/pc_%05d_vox_%.2f.pcd", ix, voxel_size);
        //    std::string front_out = front_out_buff;

        //    filterAndPrint(main_out, full_cloud_main, voxel_size);
        //    filterAndPrint(front_out, full_cloud_front, voxel_size);
        //}

    }

    //std::string fn_main = "main_out.pcd";
    std::string fn_front = "front_clouds/front_out_opt2_calibration.pcd";
    //filterAndPrint(fn_main, full_cloud_main, voxel_size);
    filterAndPrint(fn_front, full_cloud_front, voxel_size);

  return (0);
}


