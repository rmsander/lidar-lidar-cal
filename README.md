Lidar Calibration Exploratory Tools
===================================

Installation
============

In order to see the cumulative point clouds, you need to install octomap:

```
sudo apt-get install ros-melodic-octomap-server                                                
sudo apt-get install ros-melodic-octomap   
```

We also need the `pcl` library:
```
sudo apt-get install libpcl-dev
sudo apt-get install pcl-tools
```
(This works on Ubuntu 18.04. For older versions other instructions are 
necessary because I don't think `libpcl` was part of the distro repo.

clean\_csvs.py
--------------
Generates csvs for the lidar odometry with sensible header names: `time, x, y,
z, qx, qy, qz, qw`. The pose covariance is represented as `cov_pose_0` through
`cov_pose_35` in row major order (although we are still a little unclear on
the significance of this covariance.

analysis.py
-----------
Formulates and solves an optimization problem to find the maximum likelihood
estimate for the lidar-lidar transform

extract\_pose\_yaml.py
----------------------
Parses the yaml file that stores robot sensor calibration values into
homogeneous transformation matrices.

timing\_analysis.py
-------------------
Runs cross-correlation of angular velocity each each lidar frame to find time
offset between sensors. Based on our preliminary works, there doesn't seem to be
much of an offset.

relative\_pose\_processing.py
-----------------------------

A variety of utility functions to process the data. Includes retiming and
alignment, quaternion multiplication, calculating angular velocity from pairs of
quaternions, and turning the dataframes into lists of relative poses.

Octomap / PCL Tools
===================

Running `rosrun final save_octomap_cloud` generates three point clouds, one for
each lidar. These are called `pc_main.pcd, pc_front.pcd` and `pc_rear.pcd`.  We
can visualize these point clouds with `pcl_viewer` with the command:
```
pcl_viewer -multiview 1 pc_main.pcd pc_front.pcd pc_rear.pcd
```
or
```
pcl_viewer pc_main.pcd pc_front.pcd pc_rear.pcd
```
The former command displays the clouds separately, and the latter displays them
on top of each other.

Relative Pose Estimation and Analysis
===================
To estimate relative poses between the lidar frames, we use `pymanopt` for running
iterative gradient-based optimizations to estimate relative pose transformations
that minimize the distance to the observed pose transformations.  

To run this analysis, run: 
```
python3 analysis.py --weighted --use_xval --reject_threshold 100 --kfolds 10
```
Where the command-line parameters are given by:
1. `weighted`: Whether to use weighted optimization to account for noise in rotation and translation.

2. `reject_threshold`: Value of maximum ICP diagonal with which to reject a sample.

3. `kfolds`: If cross-validation is enabled, this is the number of 
cross-validation folds to use for it.

4. `use_xval`: Whether to use cross-validation when fitting the model and 
calculating relative pose estimates.

This will:

1. Estimate the relative pose transformation between:
(i) `velodyne` and `velodyne_front`, (ii) `velodyne` and `velodyne_rear`, and
(iii)) `velodyne_front` and `velodyne_rear`.  These estimates are provided an
initial guess given by the configuration file `husky4_sensors.yaml`, from which
we have extracted relative pose transformations between frames.

2. Save the final estimates of these relative pose transformations, represented
as 4 x 4 matrices, to the directory `final_estimates_unweighted_{True, False}`,
where the suffix of this directory depends on whether you use weighted estimates
to optimize the system (a flag in `analysis.py`).

3. Compute the Root Mean Squared Error (RMSE) of the relative (i) pose transformation,
(ii) rotation, and (iii) translation.  This is carried out for both the initial and 
final estimates.  These results for each pairwise combination of sensors can be
found under `analysis_results_unweighted_{True, False}`, again depending on whether
or not the samples are weighted during training.

4. Generate plots of the odometry for all three sensors.

5. Generate plots of the ICP covariance diagonal elements over time for each sensor.
These are saved in `icp_plots/` (or `icp_plots_clipped`), if the maximum value
is clipped.

6. If cross-validation is enabled, the results will be stored in a new (or existing)
directory given by `./cross_validation_folds=K`, where `K` is the number of folds.

Utility functions for running the script in `analysis.py` can be found in `analysis_utils.py`.