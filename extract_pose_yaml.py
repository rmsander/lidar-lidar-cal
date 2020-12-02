"""Script for finding relative pose between velodyne and the rest of the other sensors."""

import numpy as np
import yaml
import os
import math
import pickle

def save_transforms(f, t):
    """Function to save relative transforms to file using pickle."""
    with open(f, "wb") as pkl:
        pickle.dump(t, pkl)
        pkl.close()
        print("Relative transforms saved to: {}".format(f))

def load_transforms(f):
    """Function to load relative transforms from file."""
    with open(f, "rb") as pkl:
        my_dict = pickle.load(pkl)
        pkl.close()
    return my_dict

# Source: https://www.zacobria.com/universal-robots-knowledge-base-tech-support-
# forum-hints-tips/python-code-example-of-converting-rpyeuler-angles-to-rotation-vectorangle-axis-for-universal-robots/
def get_rotation_matrix(rpy_dict):
    """Function to convert from (r, p, y) to rotation matrix.

    Arguments:
        rpy_dict (dict): A dictionary with key-value pairs 'r': r, 'p': p, and
            'y': y denoting roll, pitch, and yaw angles, respectively.

    Returns:
        R (np.array): A 3 x 3 rotation matrix formed by multiplying the
            yaw x pitch x roll matrices.
    """

    # Get rpy
    r = rpy_dict['r']
    p = rpy_dict['p']
    y = rpy_dict['y']

    # Yaw
    yawMatrix = np.array([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y), math.cos(y), 0],
        [0, 0, 1]
    ])

    # Pitch
    pitchMatrix = np.array([
        [math.cos(p), 0, math.sin(p)],
        [0, 1, 0],
        [-math.sin(p), 0, math.cos(p)]
    ])

    # Roll
    rollMatrix = np.array([
        [1, 0, 0],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r), math.cos(r)]
    ])

    # Construct rotation matrix
    R = yawMatrix * pitchMatrix * rollMatrix
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
    print(R_t)
    T4 = np.array([0, 0, 0, 1])
    T = np.vstack((R_t, T4))

    return T


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
        rotation_matrix = get_rotation_matrix(rpy)
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

    # Now, we need to find transforms relative to velodyne
    velodyne_pose = pose_sensor_dict[
        'velodyne']  # Pose of velodyne w.r.t. world
    inv_velodyne_pose = np.linalg.inv(
        velodyne_pose)  # Pose of world w.r.t. velodyne
    assert (abs(np.linalg.det(inv_velodyne_pose @ velodyne_pose) - 1.0) < 1e-1)

    # Now we can apply transformation to the rest of the poses
    transformed_pose_sensor_dict = {}

    for sensor, pose in pose_sensor_dict.items():
        transformed_pose = inv_velodyne_pose @ pose
        transformed_pose_sensor_dict[sensor] = transformed_pose
        print("\n SENSOR: {} \n"
              "T_{}^velodyne: \n{} \n".format(sensor, sensor, transformed_pose))

    # Save relative transforms to file
    out_file = os.path.join(os.getcwd(), "relative_transforms.pkl")
    save_transforms(out_file, transformed_pose_sensor_dict)


if __name__ == '__main__':
    main()
