#!/usr/bin/python3

"""Functions and scripts to clean odometry data into clean CSVs that we later
use for calibration estimation."""

# Native Python packages
import os

# External Python packages
import pandas as pd
import matplotlib.pyplot as plt


def clean_csv(fn, ncols_icp=36):
    """Function to read an odometry CSV and clean it to make analysis more
        succinct and have more clarity.

    Parameters:
          fn (str):  String corresponding to the file name of the CSV odometry
            data we wish to load.
          ncols_icp (int):  The number of columns to iterate over in the odometric
            dataset for the ICP covariance (usually a 6x6) matrix.

    Returns:
          data_new (pd.DataFrame):  DataFrame corresponding to our cleaned
            dataset for our odometric data.
    """
    # Read in dataset and instantiate a new CSV file
    data = pd.read_csv(fn)
    data_new = pd.DataFrame()

    # Set time
    data_new['time'] = data['%time']

    # Set translation
    data_new['x'] = data['field.pose.pose.position.x']
    data_new['y'] = data['field.pose.pose.position.y']
    data_new['z'] = data['field.pose.pose.position.z']

    # Set rotation
    data_new['qx'] = data['field.pose.pose.orientation.x']
    data_new['qy'] = data['field.pose.pose.orientation.y']
    data_new['qz'] = data['field.pose.pose.orientation.z']
    data_new['qw'] = data['field.pose.pose.orientation.w']

    # Iterate over columns of odometry dataset ICP covariance matrix
    for ix in range(ncols_icp):
        data_new['cov_pose_%d' % ix] = data['field.pose.covariance%d' % ix]

    return data_new


if __name__ == '__main__':
    """Main function for reading and cleaning data."""

    # Main lidar
    main_data = clean_csv(os.path.join('data', 'main_odometry.csv'))
    main_data.to_csv(os.path.join('data', 'main_odometry_clean.csv'))

    # Front lidar
    front_data = clean_csv(os.path.join('data', 'front_odometry.csv'))
    front_data.to_csv(os.path.join('data', 'front_odometry_clean.csv'))

    # Rear lidar
    rear_data = clean_csv(os.path.join('data', 'rear_odometry.csv'))
    rear_data.to_csv(os.path.join('data', 'rear_odometry_clean.csv'))

    # Create plot for main lidar data
    main_data.cov_pose_0.plot()
    plt.show()

    # Plot front lidar data for translation
    front_data.x.plot()
    front_data.y.plot()
    front_data.z.plot()
    plt.legend(['x', 'y', 'z'])

    # Plot front lidar data for translation
    plt.figure()
    front_data.qx.plot()
    front_data.qy.plot()
    front_data.qz.plot()
    plt.plot(front_data.qx**2 + \
             front_data.qy**2 + \
             front_data.qz**2 + \
             front_data.qw**2)
    plt.legend(['qx', 'qy', 'qz', 'norm'])
    plt.show()
