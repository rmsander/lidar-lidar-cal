#!/usr/bin/python3

"""Script for finding relative pose between velodyne and the rest of the other sensors."""

import numpy as np
import yaml
import os
import pickle

from scipy.spatial.transform import Rotation as R

def save_transforms(f, t):
    """Function to save relative transforms to file using pickle."""

    # Create output directory if it doesn't already exist
    if not os.path.exists("relative_transforms"):
        os.mkdir("relative_transforms")

    # Write transforms to pkl file
    out_path = os.path.join("relative_transforms", f)
    with open(out_path, "wb") as pkl:
        pickle.dump(t, pkl)
        pkl.close()
        print("Relative transforms saved to: {}".format(out_path))

    # Write transforms to text file
    for sensor, T in t.items():
        out_path = os.path.join("relative_transforms", "{}.txt".format(sensor))
        np.savetxt(out_path, T)

def load_transforms(f):
    """Function to load relative transforms from file."""
    open_path = os.path.join("relative_transforms", f)
    with open(open_path, "rb") as pkl:
        transforms_dict = pickle.load(pkl)
        pkl.close()
    return transforms_dict


def eul2rot(rpy_dict):
    """Function to convert from (r, p, y) to rotation matrix.

    Arguments:
        rpy_dict (dict): A dictionary with key-value pairs 'r': r, 'p': p, and
            'y': y denoting roll, pitch, and yaw angles, respectively.

    Returns:
        R (np.array): A 3 x 3 rotation matrix formed by multiplying the
            yaw x pitch x roll matrices.
    """
    theta = [rpy_dict['r'], rpy_dict['p'], rpy_dict['y']]
    R = np.array([[np.cos(theta[1]) * np.cos(theta[2]),
                   np.sin(theta[0]) * np.sin(theta[1]) * np.cos(
                       theta[2]) - np.sin(theta[2]) * np.cos(theta[0]),
                   np.sin(theta[1]) * np.cos(theta[0]) * np.cos(
                       theta[2]) + np.sin(theta[0]) * np.sin(theta[2])],
                  [np.sin(theta[2]) * np.cos(theta[1]),
                   np.sin(theta[0]) * np.sin(theta[1]) * np.sin(
                       theta[2]) + np.cos(theta[0]) * np.cos(theta[2]),
                   np.sin(theta[1]) * np.sin(theta[2]) * np.cos(
                       theta[0]) - np.sin(theta[0]) * np.cos(theta[2])],
                  [-np.sin(theta[1]), np.sin(theta[0]) * np.cos(theta[1]),
                   np.cos(theta[0]) * np.cos(theta[1])]])

    return R


def get_translation(t):
    """Extracts the components of translation from dictionary
    and converts to numpy array.

    Parameters:
        t (dict):  Dictionary with key-value pairs "x": x, "y": y, "z": z.z,
            where x, y, and z are the translation relative to the world frame.

    Returns:
        t_array (np.array):  Array composed of [x, y, z] of the translation
            relative to the world frame.
    """
    return np.array([t['x'], t['y'], t['z']]).reshape((3, 1))


def construct_pose(R, t):
    """Constructs a pose given a rotation matrix and a translation vector.

    Parameters:
        R (np.array): A 3 x 3 rotation matrix corresponding to the rotation
            between the world frame and the coordinate frame this is given for.
        t (np.array): A 3 x 1 translation vector corresponding to the translation
            between the world frame and the coordinate frame this is given for.

    Returns:
        T (np.array): A 4 x 4 pose matrix describing the rotation-translation
            transformation from the world frame into the given frame.
    """
    R_t = np.hstack((R, t))
    T4 = np.array([0, 0, 0, 1])
    T = np.vstack((R_t, T4))

    return T


def construct_inv_pose(T):
    """Given a pose T, constructs an inverse pose T^{-1}."""

    # Get rotation matrix and translation vector
    R = T[:3, :3]  # 3 x 3 upper block
    t = T[:3, 3]  # 3 x 1 upper right vector

    # Get inverse elements
    R_prime = R.T  # Inverse of rotation matrix
    t_prime = -R_prime @ t.reshape((3, 1))

    # Inverse pose
    T_inv = construct_pose(R_prime, t_prime)
    return T_inv


def main():
    # Open yaml file for parsing
    with open('husky4_sensors.yaml') as f:
        my_dict = yaml.safe_load(f)  # Dump into Python dictionary
        f.close()

    # Create tables for each sensor
    sensors = []
    rpy_angles = []
    rotation_matrices = []
    translations = []
    poses = []

    # Create dictionary of poses to keep track of transformations
    pose_sensor_dict = {}

    # Get items from dictionary
    for _, value in my_dict.items():
        # Get values
        sensor = value['name']
        translation = get_translation(value['position'])
        rpy = value['orientation']
        rotation_matrix = eul2rot(rpy)
        pose = construct_pose(rotation_matrix, translation)

        # Add items to arrays
        sensors.append(sensor)
        rpy_angles.append(rpy)
        rotation_matrices.append(rotation_matrix)
        translations.append(translation)
        poses.append(pose)

        # Add to our pose-sensor dictionary
        pose_sensor_dict[sensor] = pose

    # Display sensors, positions, orientations
    for sensor, translation, rpy_angle, rotation_matrix, pose \
            in zip(sensors, translations, rpy_angles, rotation_matrices, poses):
        print("\nSENSORS: {} \n".format(sensor))
        print("TRANSLATION: {} \n".format(translation))
        print("ORIENTATIONS: {} \n".format(rpy_angle))
        print("ROTATION MATRICES: \n {} \n".format(rotation_matrix))
        print("POSE: \n {} \n".format(pose))

        inverse_mat = np.linalg.inv(rotation_matrix)
        inverse_trans = -inverse_mat @ translation
        rotation_inv = R.from_matrix(inverse_mat)
        inverse_quat = rotation_inv.as_quat()
        print("INVERSE QUATERNION: {} \n".format(inverse_quat))
        print("INVERSE TRANSLATION: {} \n".format(inverse_trans))

    # Now, we need to find transforms relative to velodyne
    velodyne_pose = pose_sensor_dict['velodyne']  # Pose of velodyne w.r.t. world
    inv_velodyne_pose = construct_inv_pose(velodyne_pose)
    print("INV CHECK: {}".format(velodyne_pose @ inv_velodyne_pose))
    assert (abs(np.linalg.det(inv_velodyne_pose @ velodyne_pose) - 1.0) < 1e-1)

    # Now we can apply transformation to the rest of the poses
    transformed_pose_sensor_dict = {}

    for sensor, pose in pose_sensor_dict.items():
        transformed_pose = inv_velodyne_pose @ pose
        transformed_pose_sensor_dict[sensor] = transformed_pose
        print("\n SENSOR: {} \n"
              "T_{}^velodyne: \n{} \n".format(sensor, sensor, transformed_pose))
        print("DET: {}".format(np.linalg.det(transformed_pose_sensor_dict[sensor])))

    # As a sanity check, compute front-to-rear
    velodyne_front_pose = pose_sensor_dict['velodyne_front']
    velodyne_rear_pose = pose_sensor_dict['velodyne_rear']
    inv_velodyne_front_pose = construct_inv_pose(velodyne_front_pose)
    transformed_pose_sensor_dict["direct_front_rear"] = inv_velodyne_front_pose @ velodyne_rear_pose

    # Save relative transforms to file
    out_file = "relative_transforms.pkl"
    save_transforms(out_file, transformed_pose_sensor_dict)





if __name__ == '__main__':
    main()
