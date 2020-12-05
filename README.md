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
