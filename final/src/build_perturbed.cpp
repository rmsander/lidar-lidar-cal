#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <stdlib.h>
#include <math.h>

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
    // sample_offset n_samples scale_offset scale1 scale2 ... scalen

    int sample_offset = atoi(argv[1]);
    int n_rot_perturb_each = atoi(argv[2]);
    int scale_offset = atoi(argv[3]);
    //std::vector<double> rot_perturb_scales{0.01, 0.04, 0.1, 0.2, 0.5};
    std::vector<double> rot_perturb_scales;
    for (auto ix = 4; ix < argc; ++ix) {
        rot_perturb_scales.push_back(std::stof(argv[ix]));
    }

    int start_scan = 0;
    int end_scan = 150;
    //int n_scans = 589;
    //if (argc > 2) {
    //    start_scan = atoi(argv[1]);
    //    end_scan = atoi(argv[2]);
    //}
    //double voxel_size = 0.03;
    //if (argc > 3) {
    //    voxel_size = std::stof(argv[3]);
    //}
    double voxel_size = 0.05;

    Eigen::Matrix4d T_front_main = readMatrix4("velodyne_front.txt");

    //std::vector<std::vector<double>> fitness_vecs;
    //fitness_vecs.resize(rot_perturb_scales.size());

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
        //fitness_vecs[scale_index].resize(n_rot_perturb_each);
        for (auto sample_index = 0; sample_index < n_rot_perturb_each; ++sample_index) {
            std::cout << "Scale ix, sample ix: " << scale_index << "," << sample_index << std::endl;
            double r = rot_perturb_scales[scale_index] * static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
            double p = rot_perturb_scales[scale_index] * static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
            double y = rot_perturb_scales[scale_index] * static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

            Eigen::Matrix4d perturbation_matrix = rpyToHomog(r, p, y);

            pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
            for (auto ix = start_scan; ix < end_scan; ++ix) {

                //char buff1[100];
                //snprintf(buff1, sizeof(buff1), "main_lidar/pc_%05d.pcd", ix);
                //std::string pc_name = buff1;

                //char buff_front[100];
                //snprintf(buff_front, sizeof(buff_front), "front_lidar/pc_%05d.pcd", ix);
                //std::string pc_name_front = buff_front;

                //char buff_pose[100];
                //snprintf(buff_pose, sizeof(buff_pose), "main_lidar_pose/pose_%05d.txt", ix);
                //std::string pose_name = buff_pose;
                //Eigen::Matrix4d T = readMatrix4(pose_name.c_str());

                //// Main cloud
                //pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
                //loadCloud(pc_name, temp_cloud);

                //pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
                //temp_cloud = main_clouds[ix];
                pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
                temp_cloud_front = front_clouds[ix];
                Eigen::Matrix4d T = transforms[ix];
                
                // Executing the transformation
                //pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
                //pcl::transformPointCloud (*temp_cloud, *transformed_cloud, T);


                //(*full_cloud_main) += (*transformed_cloud);
                //std::cout << full_cloud_main->size() << std::endl;

                //// Front Cloud
                //pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud_front (new pcl::PointCloud<pcl::PointXYZ>);
                //loadCloud(pc_name_front, temp_cloud_front);

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


