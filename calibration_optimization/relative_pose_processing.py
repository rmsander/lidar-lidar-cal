#!/usr/bin/python3

"""Contains functions and scripts for cleaning csvs and calculating relative
poses between times."""

# Native python package
import itertools

# External packages
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def retime_data(df):
    """Resamples the dataframe to 10hz (and aligns to 0.1 second intervals).

    Parameters:
        df (pd.DataFrame):  DataFrame containing the odometric data of interest
            to be aligned to 0.1 seconds.0

    Returns:
        df (pd.DataFrame): DataFrame modified for time alignment.
    """
    # Retime and realign to 10 Hz
    df['time'] = pd.to_datetime(df['time'], unit='ns')
    df = df.set_index('time')
    df = df.resample('.1S').mean()
    df = df.interpolate()
    return df


def quatmul(q1, q2):
    """Performs quaternion multiplication of two four-component quaternions.

    Parameters:
        q1 (tuple or list):  The first 4-vector quaternion to be multiplied.
        q2 (tuple or list):  The second 4-vector quaternion to be multiplied.

    Returns:
        q3 (tuple or list):  The product of the two quaternions, corresponding
            to another 4-vector quaternion.
    """
    # Extract quaternion components
    a, b, c, d = q1
    e, f, g, h = q2

    # Calculate each quaternion component of the product
    w = a * e - b * f - c * g - d * h  # First component
    x = b * e + a * f + c * h - d * g  # Second component
    y = a * g - b * h + c * e + d * f  # Third component
    z = a * h + b * g - c * f + d * e  # Fourth component

    # Return the product as four product coefficients
    return [w, x, y, z]


def quatmul_delta_pandas(df, df_shift):
    """Function that computes the the rotation difference between two
    quaternions, expressed as a quaternion.

    Parameters:
        df (pd.DataFrame): DataFrame corresponding to odometry data
            for the lidar frame of interest.
        df_shift (pd.DataFrame): DataFrame corresponding to odometry data
            for the lidar frame of interest, shifted in time.  This shift is
            invoked through the 'retime_data' function specified below.

    Returns:
        q_diff (list):  A 4-element list corresponding to a relative rotation
            between two quaternions.  This quantity in turn is expressed as
            a quaternion.
    """
    # Extract elements of q2 (quaternion 2)
    a = df_shift.qw
    b = df_shift.qx
    c = df_shift.qy
    d = df_shift.qz

    # Extract elements of q1 (quaternion 1)
    e = -df.qw  # This inverts q1, by definition of quaternions
    f = df.qx
    g = df.qy
    h = df.qz

    # Compute the relative rotation between the two quaternions
    dw = (a * e - b * f - c * g - d * h).clip(-1, 1)
    dx = b * e + a * f + c * h - d * g
    dy = a * g - b * h + c * e + d * f
    dz = a * h + b * g - c * f + d * e

    # Return the relative rotation
    return [dw, dx, dy, dz]

def calc_rel_poses(df):
    """Function that computes the relative pose transformation for a dataframe,
    for each sequential pair of points in time.

    Parameters:
        df (pd.DataFrame): DataFrame corresponding to odometry data
            for the lidar frame of interest.  This is prior to any processing.

    Returns:
        df (pd.DataFrame): DataFrame corresponding to odometry data
            for the lidar frame of interest.  This is after relative pose processing
            has been finished.
        T_homg_list (list):  A list of relative pose transformations for a single
            lidar at two separate points in time.  These poses are used in our
            nonlinear optimization routine between two (or more) sensors.
    """
    # Extract translation
    x = df.x.values
    y = df.y.values
    z = df.z.values

    # Extract rotation, expressed as a quaternion
    dqw = df.dqw.values
    dqx = df.dqx.values
    dqy = df.dqy.values
    dqz = df.dqz.values

    # Initiative delta values
    dx_vals = [0]
    dy_vals = [0]
    dz_vals = [0]
    T_homg_list = []


    # Loop over rows of the DataFrame, a.k.a. odometry measurements
    for ix in range(len(x) - 1):

        # Initialize a relative pose transformation
        T_homg = np.zeros((4, 4))

        # Compute the rotation matrix from the given quaternion
        rot = R.from_quat([dqx[ix], dqy[ix], dqz[ix], dqw[ix]]).as_matrix()

        # Compute shifts
        dx = x[ix + 1] - x[ix]
        dy = y[ix + 1] - y[ix]
        dz = z[ix + 1] - z[ix]

        # Append shifts
        dx_vals.append(dx)
        dy_vals.append(dy)
        dz_vals.append(dz)

        # Finally, set the values of the relative pose
        T_homg[:3, :3] = rot
        T_homg[0, 3] = dx
        T_homg[1, 3] = dy
        T_homg[2, 3] = dz
        T_homg[3, 3] = 1
        T_homg_list.append(T_homg)

    # Create new columns in the DataFrame
    df['dx'] = np.array(dx_vals)
    df['dy'] = np.array(dy_vals)
    df['dz'] = np.array(dz_vals)

    return (df, T_homg_list)


def process_df(fn):
    """Given a csv filename from the rosbag, compute the relative rotation information
    from the odometry information.  Note the following:

        (i) dqw is the w component of the relative rotation quaternion between
            timesteps.
        (ii) dtheta is the angle difference between timesteps
            (e.g. in an axis-angle interpretation). If

    Parameters:
        fn (str):  String corresponding to the file path of the relevant
            CSV file containing odometry data that we use to estimate relative
            poses between.

    Returns:
         df (pd.DataFrame):  DataFrame corresponding to odometry data
            for the lidar frame of interest.  This DataFrame contains new columns,
            such as the differences in the quaternion components between frames
            for a given lidar sensor.
    """
    # Read and re-time CSV file
    df = pd.read_csv(fn)
    df = retime_data(df)

    # Compute relative rotation between all pairs of measurements
    [dqw, dqx, dqy, dqz] = quatmul_delta_pandas(df, df.shift())

    # Add relative quaternion components between frames to the odometry dataset
    df['dqw'] = dqw
    df['dqx'] = dqx
    df['dqy'] = dqy
    df['dqz'] = dqz

    # Add relative angles between framers to the odometry dataset
    df['dtheta'] = 2 * np.arccos(df.dqw) - 2 * np.pi  # Map 1
    df['dtheta'][df['dtheta'] < -np.pi] += 2 * np.pi  # Map 2 (result is [0, 2pi])
    df['omega'] = df.dtheta / 0.1  # Since we run this at 10 Hz

    return df

def align_df(pd_objects):
    """This function performs an inner join on all dataframes in the list pd_objects.
    It applies align on all combinations of the list of pd objects.

    Parameters:
        pd_objects (list or tuple):  List/tuple of odometry DataFrames that are
            to be aligned.

    Returns:
        pd_objects (list or tuple):  List/tuple of odometry DataFrames that are
            now aligned.
    """
    # Iterate over different odometry DataFrames
    for (i, j) in itertools.combinations(range(len(pd_objects)), 2):
        (pd_objects[i], pd_objects[j]) = pd_objects[i].align(pd_objects[j], 'inner')  # Perform inner join

    # Iterate over different odometry DataFrames
    for i in range(len(pd_objects)):
        pd_objects[i] = pd_objects[i].fillna(0)
        pd_objects[i] = pd_objects[i].drop(pd_objects[i].index[0])
        pd_objects[i] = pd_objects[i].drop(pd_objects[i].index[-1])

    return tuple(pd_objects)

