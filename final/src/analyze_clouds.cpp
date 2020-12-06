#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

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

double pairwiseFitnessICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2) {
    // Test main <-> front calibration
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    // Set the input source and target
    icp.setInputCloud (cloud1);
    icp.setInputTarget (cloud2);
     
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher
    //distances will be ignored) 
    icp.setMaxCorrespondenceDistance (0.05);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations (50);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon (1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon (1);
     
    pcl::PointCloud<pcl::PointXYZ> cloud_source_registered;
    // Perform the alignment
    icp.align (cloud_source_registered);
     
    // Obtain the transformation that aligned cloud_source to cloud_source_registered
    //Eigen::Matrix4f transformation = icp.getFinalTransformation ();

    //std::cout << transformation << std::endl;

    double fitness = icp.getFitnessScore(2);
    return fitness;

}
//  for (const auto& point: *cloud)
//    std::cout << "    " << point.x
//              << " "    << point.y
//              << " "    << point.z << std::endl;

int main (int argc, char** argv) {
    std::string prefix;
    if (argc == 1) {
        std::string prefix = "prefix";
        std::string name_main = prefix + "pc_main.pcd";
        std::string name_front = prefix + "pc_front.pcd";
        std::string name_rear = prefix + "pc_rear.pcd";

        pcl::PointCloud<pcl::PointXYZ>::Ptr main_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        loadCloud(name_main, main_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        loadCloud(name_front, front_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr rear_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        loadCloud(name_rear, rear_cloud);
        double fitness_mf = pairwiseFitnessICP(front_cloud, main_cloud);
        double fitness_mr = pairwiseFitnessICP(rear_cloud, main_cloud);
        std::cout << "Fitness (mf, mr): " << fitness_mf << "," << fitness_mr << std::endl;
        return 0;
    } 

    std::ofstream outfile_mf;
    outfile_mf.open ("output_mf.txt");
    std::ofstream outfile_mr;
    outfile_mr.open ("output_mr.txt");
    for (auto ix = 1; ix < argc; ++ix) {
        std::string prefix = argv[ix];

        std::string name_main = prefix + "pc_main.pcd";
        std::string name_front = prefix + "pc_front.pcd";
        std::string name_rear = prefix + "pc_rear.pcd";

        pcl::PointCloud<pcl::PointXYZ>::Ptr main_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        loadCloud(name_main, main_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        loadCloud(name_front, front_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr rear_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        loadCloud(name_rear, rear_cloud);
        double fitness_mf = pairwiseFitnessICP(front_cloud, main_cloud);
        double fitness_mr = pairwiseFitnessICP(rear_cloud, main_cloud);

        outfile_mf << fitness_mf << std::endl;
        outfile_mr << fitness_mr << std::endl;
        
        std::cout << "Fitness (mf, mr): " << fitness_mf << "," << fitness_mr << std::endl;
    }
    outfile_mf.close();
    outfile_mr.close();


  return (0);
}


