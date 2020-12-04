# relative_pose_processing.py -- Take clean csvs and calculate relative poses between times
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.transform import Rotation as R


def retime_data(df):
    # Resamples the dataframe to 10hz (and aligns to .1 s intervals)
    df['time'] = pd.to_datetime(df['time'], unit='ns')
    df = df.set_index('time')
    df = df.resample('.1S').mean()
    df = df.interpolate()
    return df


def quatmul(q1, q2):
    # multiply two quaternions, q1*q2
    a, b, c, d = q1
    e, f, g, h = q2

    w = a * e - b * f - c * g - d * h
    x = b * e + a * f + c * h - d * g
    y = a * g - b * h + c * e + d * f
    z = a * h + b * g - c * f + d * e
    return [w, x, y, z]


def quatmul_delta_pandas(df, df_shift):
    # Compute the the rotation difference between two quaternions, expressed as a quaternion
    # [dw, dx, dy, dz] * df = df_shift
    # q = df_shift * df^{-1}

    a = df_shift.qw
    b = df_shift.qx
    c = df_shift.qy
    d = df_shift.qz

    e = -df.qw  # This inverts q1
    f = df.qx
    g = df.qy
    h = df.qz

    dw = (a * e - b * f - c * g - d * h).clip(-1, 1)
    dx = b * e + a * f + c * h - d * g
    dy = a * g - b * h + c * e + d * f
    dz = a * h + b * g - c * f + d * e
    return [dw, dx, dy, dz]

def calc_rel_poses(df):
    # Computes the relative pose transformation for a dataframe, for each sequential pair of times
    x = df.x.values
    y = df.y.values
    z = df.z.values
    dqw = df.dqw.values
    dqx = df.dqx.values
    dqy = df.dqy.values
    dqz = df.dqz.values
    T_homg_list = []
    for ix in range(len(x) - 1):
        T_homg = np.zeros((4, 4))
        # rot = quaternion_matrix([dqw, dqx, dqy, dqz])
        rot = R.from_quat([dqx[ix], dqy[ix], dqz[ix], dqw[ix]]).as_matrix()
        dx = x[ix + 1] - x[ix]
        dy = y[ix + 1] - y[ix]
        dz = z[ix + 1] - z[ix]
        T_homg[:3, :3] = rot
        T_homg[0, 3] = dx
        T_homg[1, 3] = dy
        T_homg[2, 3] = dz
        T_homg[3, 3] = 1
        T_homg_list.append(T_homg)

    return T_homg_list


def process_df(fn):
    # Given a csv filename from the rosbag, compute the relative rotation information
    # dqw is the w component of the relative rotation quaternion between timesteps
    # dtheta is the angle difference between timesteps (e.g. in an axis-angle interpretation)
    df = pd.read_csv(fn)
    df = retime_data(df)
    [dqw, dqx, dqy, dqz] = quatmul_delta_pandas(df, df.shift())
    df['dqw'] = dqw
    df['dqx'] = dqx
    df['dqy'] = dqy
    df['dqz'] = dqz

    df['dtheta'] = 2 * np.arccos(df.dqw) - 2 * np.pi
    df['dtheta'][df['dtheta'] < -np.pi] += 2 * np.pi
    df['omega'] = df.dtheta / .1

    return df

def align_df(pd_objects):
    # performs an inner join on all dataframes in the list pd_objects
    """apply align on all combinations of the list of pd objects"""
    for (i, j) in itertools.combinations(range(len(pd_objects)), 2):
        (pd_objects[i], pd_objects[j]) = pd_objects[i].align(pd_objects[j], 'inner')

    for i in range(len(pd_objects)):
        pd_objects[i] = pd_objects[i].fillna(0)
        pd_objects[i] = pd_objects[i].drop(pd_objects[i].index[0])
        pd_objects[i] = pd_objects[i].drop(pd_objects[i].index[-1])
    return tuple(pd_objects)

