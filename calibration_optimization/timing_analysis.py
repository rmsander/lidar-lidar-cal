#!/usr/bin/python3

"""Script to compute the timing for our odometry data.  Computes cross-correlation
between signals over time."""

# Native Python packages
import os

# External packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import relative_pose_processing


def compute_xcorr(odom_1, odom_2, type_1="main", type_2="front"):
    """Function to compute the cross-correlation between two sets of odometric
    measurements according to the angle omega.

    Parameters:
        odom_1 (df):  Odometry dataframe corresponding to odometry data for one
            of the lidar sensors we wish to find relative transform of relative
            to another lidar.
        odom_2 (df):  Odometry dataframe corresponding to odometry data for the
            other of the lidar sensors we wish to find relative transform of relative
            to another lidar.
    """
    # Compute cross-correlation
    xcorr = scipy.signal.correlate(odom_1.omega, odom_2.omega)

    # Compute time delay
    print('Time delay:', np.argmax(xcorr) - len(odom_1.omega))

    # Plot cross-correlation
    plt.plot(xcorr)
    plt.xlabel("Timesteps (0.1 seconds)")
    plt.ylabel("Cross-correlation")
    plt.title("Cross-correlation between odometric measurements.")
    plt.savefig(os.path.join("plots", "xcorr_{}_{}.png".format(type_1, type_2)))

    # Create figures for angle between two odometric measurements
    plt.figure()
    odom_1.omega.plot()  # Plot omega for odom 1
    odom_2.omega.plot()  # Plot omega for odom 2
    plt.xlabel("Timesteps (0.1 seconds)")
    plt.ylabel("Omega")
    plt.title("Angle Between {} and {} Odometry Over Time".format(type_1, type_2))
    plt.savefig(os.path.join("plots", "odom_pair_{}_{}.png".format(type_1, type_2)))


def main():
    """Main function to perform timing analysis and cross-correlation computation
    between two sets of odometric data."""
    # Preprocess data
    main_odometry = relative_pose_processing.process_df(os.path.join(
        'data', 'main_odometry_clean.csv'))
    front_odometry = relative_pose_processing.process_df(os.path.join(
        'data', 'front_odometry_clean.csv'))
    rear_odometry = relative_pose_processing.process_df(os.path.join(
        'data', 'rear_odometry_clean.csv'))

    # Perform data alignment
    (main_aligned, front_aligned, rear_aligned) = \
        relative_pose_processing.align_df([main_odometry,
                                           front_odometry,
                                           rear_odometry])

    # Compute cross correlation between the three different frames
    odom_pairs = [(main_aligned, front_aligned),
                  (main_aligned, rear_aligned),
                  (front_aligned, rear_aligned)]
    type_pairs = [("main", "front"), ("main", "rear"), ("front", "rear")]

    # Iterate over pairs
    for odom_pair, type_pair in zip(odom_pairs, type_pairs):
        odom_1, odom_2 = odom_pair  # Extract odometry
        type_1, type_2 = type_pair  # Extract types
        compute_xcorr(odom_1, odom_2, type_1=type_1, type_2=type_2)


if __name__ == '__main__':
    main()
