#!/usr/bin/python3

"""Functions and script for finding relative pose between velodyne
and the rest of the other sensors.  These relative transforms are saved for
initial guesses for the optimization carried out in analysis.py."""

# Native python imports
import yaml
import os
import pickle

# External packages
import numpy as np
from scipy.spatial.transform import Rotation as R

def save_transforms(f, t):
    """Function to save relative transforms to file using pickle.

    Parameters:
        f (string):  String corresponding to the name of the file to be saved
            for the transforms.
        t (np.array): A 4x4 matrix corresponding to a pose estimate beteen two
            lidars that we save to file.
    """
    # Create output directory if it doesn't already exist
    if not os.path.exists("relative_transforms"):
        os.mkdir("relative_transforms")

    # Write transforms to pkl file
    with open(f, "wb") as pkl:
        pickle.dump(t, pkl)
        pkl.close()
        print("Relative transforms saved to: {}".format(f))

    # Write transforms to text file
    for sensor, T in t.items():
        out_path = os.path.join("relative_transforms", "{}.txt".format(sensor))
        np.savetxt(out_path, T)

def load_transforms(f):
    """Function to load relative transforms from file.

    Parameters:
        f (string):  String corresponding to the name of the file to be saved
            for the transforms.

    Returns:
        transforms_dict (dict):  A dictionary to load from file, in which each key
            corresponds to the name of a relative transform and each value
            corresponds to a numpy array corresponding to that transform.
    """
    open_path = os.path.join(f)
    if os.path.exists(open_path):  # Check if path exists
        with open(open_path, "rb") as pkl:
            transforms_dict = pickle.load(pkl)
            pkl.close()
        return transforms_dict

    else:  # Notify user path doesn't exist
        print("File does not exist")


def eul2rot(rpy_dict):
    """Function to convert from (r, p, y) to rotation matrix.

    Arguments:
        rpy_dict (dict): A dictionary with key-value pairs 'r': r, 'p': p, and
            'y': y denoting roll, pitch, and yaw angles, respectively.

    Returns:
        R (np.array): A 3 x 3 rotation matrix formed by multiplying the
            yaw x pitch x roll matrices.
    """
    # Concatenate theta vector
    theta = [rpy_dict['r'], rpy_dict['p'], rpy_dict['y']]

    # Create rotation matrix
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
    """Given a pose T, constructs an inverse pose T^{-1}.

    Parameters:
        T (np.array):  A 4x4 matrix corresponding to the pose we would like to
            invert.  The upper 3x3 block of this matrix corresponds to the
            rotation, and the last column corresponds to the translation vector
            with homogeneous coordinates.

    Returns:
        T_inv (np.array):  A pose corresponding to the inverted pose of the input.
            For instance, if we have T^{A}_{B}, then T_inv = T^{B}_{A}.  The
            upper 3x3 block of this matrix corresponds to the inverted rotation,
            and the last column corresponds to the inverted rotation multiplied
            by translation vector with homogeneous coordinates.
    """
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
    """Main function for extracting poses from the husky sensor."""
    # Open yaml file for parsing
    husky_sensor_fpath = os.path.join('data', 'husky4_sensors.yaml')
    with open(husky_sensor_fpath, "r") as f:
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
        sensor = value['name']  # Get sensor name
        translation = get_translation(value['position'])  # Position vector
        rpy = value['orientation']  # Rotation angles as rpy
        rotation_matrix = eul2rot(rpy)  # 3x3 Rotation matrix
        pose = construct_pose(rotation_matrix, translation)  # 4x4 pose matrix

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

        inverse_mat = np.linalg.inv(rotation_matrix)  # Inverse rotation matrix
        inverse_trans = -inverse_mat @ translation  # Inverse translation
        rotation_inv = R.from_matrix(inverse_mat)  # Rotation matrix inverse
        inverse_quat = rotation_inv.as_quat()  # Inverse quaternion
        print("INVERSE QUATERNION: {} \n".format(inverse_quat))
        print("INVERSE TRANSLATION: {} \n".format(inverse_trans))

    # Now, we need to find transforms relative to velodyne
    velodyne_pose = pose_sensor_dict['velodyne']  # Pose of velodyne w.r.t. world
    inv_velodyne_pose = construct_inv_pose(velodyne_pose)
    print("INV CHECK: {}".format(velodyne_pose @ inv_velodyne_pose))  # Should be 4x4 identity matrix
    assert (abs(np.linalg.det(inv_velodyne_pose @ velodyne_pose) - 1.0) < 1e-1)

    # Now we can apply transformation to the rest of the poses
    transformed_pose_sensor_dict = {}

    for sensor, pose in pose_sensor_dict.items():  # Iterate jointly over sensors and poses
        transformed_pose = inv_velodyne_pose @ pose
        transformed_pose_sensor_dict[sensor] = transformed_pose
        print("\n SENSOR: {} \n"
              "T_{}^velodyne: \n{} \n".format(sensor, sensor, transformed_pose))
        print("DET: {}".format(np.linalg.det(transformed_pose_sensor_dict[sensor])))

    # As a sanity check, directly compute front-to-rear transformation
    velodyne_front_pose = pose_sensor_dict['velodyne_front']
    velodyne_rear_pose = pose_sensor_dict['velodyne_rear']
    inv_velodyne_front_pose = construct_inv_pose(velodyne_front_pose)
    transformed_pose_sensor_dict["direct_front_rear"] = inv_velodyne_front_pose @ velodyne_rear_pose

    # Save relative transforms to file
    out_file = os.path.join('data', 'relative_transforms.pkl')
    save_transforms(out_file, transformed_pose_sensor_dict)


if __name__ == '__main__':
    main()
