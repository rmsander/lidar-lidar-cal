Lidar Calibration Exploratory Tools
===================================

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
