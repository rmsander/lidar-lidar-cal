#!/usr/bin/python3

import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R

import autograd.numpy as np

from pymanopt.manifolds import Euclidean, Rotations, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions

import relative_pose_processing
from extract_pose_yaml import load_transforms


# Specify odometry CSV file paths
MAIN_ODOM_CSV = 'main_odometry_clean.csv'
FRONT_ODOM_CSV = 'front_odometry_clean.csv'
REAR_ODOM_CSV = 'rear_odometry_clean.csv'

# Specify path for relative transformations between poses
PKL_POSES_PATH = 'relative_transforms.pkl'

def compute_weights(R, t):
    """Computes the covariance of the rotation matrix and the standard
    deviation of translation matrices to weight the samples in the optimization
    objective."""
    pass

def parse_icp_cov(odom_df):
    """Function to parse the ICP covariance matrix for the given input df.

    Paramaters:
        odom_df (pd.DataFrame):  DataFrame containing the (cleaned) odometric
            data for one of the lidars.

    Returns:
        cov_by_frame (dict):  Dictionary where each key is a frame number, and
            each value is a corresponding ICP 6x6 matrix for that specific frame.
    """
    cov_columns = odom_df.filter(regex=("cov_pose_*"))
    cov_by_frame = {i: 0 for i in range(cov_columns.shape[0])}
    for i, row in enumerate(cov_columns.iterrows()):
        cov_by_frame[i] = np.array(row[1:]).reshape((6, 6))  # Reshape into 6 x 6 covariance matrix
    return cov_by_frame




def cost_main_front(X):
    """Function to compute the cost between two sets of poses given by the
    lists A and B.  This cost function is passed in as a parameter to the
    manifold SE(3) manifold optimization.

    This cost function is computed between the main and front poses.

    Parameters:
        X (tuple): A tuple containing a 3x3 rotation matrix estimate given by the
            R, and a 3 x1 translation vector t.  The manifold optimization is
            carried out in SE(3).

    Returns:
        cost (float):  The total cost computed over the sets of poses.
    """
    # Convert lists to autograd arrays
    A = np.array(front_rel_poses)
    B = np.array(main_rel_poses)
    # Note that we are optimizing to compute T^{B}_{A}

    # Initialize cost
    cost = 0

    # Estimates for rotation (R) and translation (t)
    R, t = X

    # Construct 4 x 4 pose manually
    R_t = np.hstack((R, t.reshape((3, 1))))  # 3 x 4 upper block
    T4 = np.array([0, 0, 0, 1])  # Use 1 in last entry for homogeneous coordinates
    Tab = np.vstack((R_t, T4))  # Merge 3 x 4 with 1 x 4 to get pose

    # Sum the cost over all poses
    for ix in range(len(A)):
        a = A[ix]
        b = B[ix]
        cost += np.square(np.linalg.norm(b @ Tab @ np.linalg.inv(a) - Tab))

    return cost

def cost_main_rear(X):
    """Function to compute the cost between two sets of poses given by the
    lists A and B.  This cost function is passed in as a parameter to the
    manifold SE(3) manifold optimization.

    This cost function is computed between the main and rear poses.

    Parameters:
        X (tuple): A tuple containing a 3x3 rotation matrix estimate given by the
            R, and a 3x1 translation vector t.  The manifold optimization is
            carried out in SE(3).

    Returns:
        cost (float):  The total cost computed over the sets of poses.
    """
    # Convert lists to autograd arrays
    A = np.array(rear_rel_poses)
    B = np.array(main_rel_poses)
    # Note that we are optimizing to compute T^{B}_{A}

    # Initialize cost
    cost = 0

    # Estimates for rotation (R) and translation (t)
    R, t = X

    # Construct 4 x 4 pose manually
    R_t = np.hstack((R, t.reshape((3, 1))))  # 3 x 4 upper block
    T4 = np.array([0, 0, 0, 1])  # Use 1 in last entry for homogeneous coordinates
    Tab = np.vstack((R_t, T4))  # Merge 3 x 4 with 1 x 4 to get pose

    # Sum the cost over all poses
    for ix in range(len(A)):
        a = A[ix]
        b = B[ix]
        cost += np.square(np.linalg.norm(b @ Tab @ np.linalg.inv(a) - Tab))

    return cost


def cost_front_rear(X):
    """Function to compute the cost between two sets of poses given by the
    lists A and B.  This cost function is passed in as a parameter to the
    manifold SE(3) manifold optimization.

    This cost function is computed between the front and rear poses.

    Parameters:
        X (tuple): A tuple containing a 3x3 rotation matrix estimate given by the
            R, and a 3x1 translation vector t.  The manifold optimization is
            carried out in SE(3).

    Returns:
        cost (float):  The total cost computed over the sets of poses.
    """
    # Convert lists to autograd arrays
    B = np.array(rear_rel_poses)
    A = np.array(front_rel_poses)
    # Note that we are optimizing to compute T^{B}_{A}

    # Initialize cost
    cost = 0

    # Estimates for rotation (R) and translation (t)
    R, t = X

    # Construct 4 x 4 pose manually
    R_t = np.hstack((R, t.reshape((3, 1))))  # 3 x 4 upper block
    T4 = np.array([0, 0, 0, 1])  # Use 1 in last entry for homogeneous coordinates
    Tab = np.vstack((R_t, T4))  # Merge 3 x 4 with 1 x 4 to get pose

    # Sum the cost over all poses
    for ix in range(len(A)):
        a = A[ix]
        b = B[ix]
        cost += np.square(np.linalg.norm(b @ Tab @ np.linalg.inv(a) - Tab))

    return cost

# Extract and process the CSVs
main_odometry = relative_pose_processing.process_df(MAIN_ODOM_CSV)
front_odometry = relative_pose_processing.process_df(FRONT_ODOM_CSV)
rear_odometry = relative_pose_processing.process_df(REAR_ODOM_CSV)

# Process poses
(main_aligned, front_aligned, rear_aligned) = relative_pose_processing.align_df([main_odometry, front_odometry, rear_odometry])

# Get ICP covariance matrices
main_icp = parse_icp_cov(main_odometry)
front_icp = parse_icp_cov(front_odometry)
rear_icp = parse_icp_cov(rear_odometry)

# Calculate relative poses
main_rel_poses = relative_pose_processing.calc_rel_poses(main_aligned)
front_rel_poses = relative_pose_processing.calc_rel_poses(front_aligned)
rear_rel_poses = relative_pose_processing.calc_rel_poses(rear_aligned)

# Optimization (1) Instantiate a manifold
translation_manifold = Euclidean(3)  # Translation vector
so3 = Rotations(3)  # Rotation matrix
manifold = Product((so3, translation_manifold))  # Instantiate manifold

# Get initial guesses for our estimations
initial_poses = {}
if os.path.exists(PKL_POSES_PATH):  # Check to make sure path exists
    transforms_dict = load_transforms(PKL_POSES_PATH)  # Loads relative transforms

# Now get initial guesses from the relative poses
initial_guess_main_front = transforms_dict["velodyne_front"]  # Get relative transform from main to front (T^{V}_{F})
initial_guess_main_rear = transforms_dict["velodyne_rear"]    # Get relative transform from front to main T^{V}_{B})
initial_guess_front_rear = np.linalg.inv(initial_guess_main_front) @ initial_guess_main_rear  # Get relative transform from front to rear T^{B}_{W})

print("INITIAL GUESS MAIN FRONT: \n {} \n".format(initial_guess_main_front))
print("INITIAL GUESS MAIN REAR: \n {} \n".format(initial_guess_main_rear))
print("INITIAL GUESS FRONT REAR: \n {} \n".format(initial_guess_front_rear))

# Get rotation matrices for initial guesses
R0_main_front, t0_main_front = initial_guess_main_front[:3, :3], initial_guess_main_front[:3, 3]
X0_main_front = (R0_main_front, t0_main_front)
print("INITIAL GUESS MAIN FRONT: \n R0: \n {} \n\n t0: \n {} \n".format(R0_main_front, t0_main_front))

R0_main_rear, t0_main_rear = initial_guess_main_rear[:3, :3], initial_guess_main_rear[:3, 3]
X0_main_rear = (R0_main_rear, t0_main_rear)
print("INITIAL GUESS MAIN REAR: \n R0: \n {} \n\n t0: \n {} \n".format(R0_main_rear, t0_main_rear))

R0_front_rear, t0_front_rear = initial_guess_front_rear[:3, :3], initial_guess_front_rear[:3, 3]
X0_front_rear = (R0_front_rear, t0_front_rear)
print("INITIAL GUESS FRONT REAR: \n R0: \n {} \n\n t0: \n {} \n".format(R0_front_rear, t0_front_rear))

# Carry out optimization for main-front homogeneous transformations
problem_main_front = Problem(manifold=manifold, cost=cost_main_front)  # (2a) Compute the optimization between main and front
solver_main_front = SteepestDescent()  # (3) Instantiate a Pymanopt solver
#Xopt_main_front = solver_main_front.solve(problem_main_front, x=X0_main_front)
print("Initial Guess for Main-Front Transformation: \n {}".format(initial_guess_main_front))
#print("Optimal solution between main and front reference frames: \n {}".format(Xopt_main_front))

# Carry out optimization for main-rear homogeneous transformations
problem_main_rear = Problem(manifold=manifold, cost=cost_main_rear)  # (2a) Compute the optimization between main and front
solver_main_rear = SteepestDescent()  # (3) Instantiate a Pymanopt solver
Xopt_main_rear = solver_main_rear.solve(problem_main_rear, x=X0_main_rear)
print("Initial Guess for Main-Rear Transformation: \n {}".format(initial_guess_main_rear))
print("Optimal solution between main and rear reference frames: \n {}".format(Xopt_main_rear))

# Carry out optimization for front-rear homogeneous transformations
problem_front_rear = Problem(manifold=manifold, cost=cost_front_rear)  # (2a) Compute the optimization between main and front
solver_front_rear = SteepestDescent()  # (3) Instantiate a Pymanopt solver
Xopt_front_rear = solver_front_rear.solve(problem_front_rear, x=X0_front_rear)
print("Initial Guess for Front-Rear Transformation: \n {}".format(initial_guess_front_rear))
print("Optimal solution between front and rear reference frames: \n {}".format(Xopt_front_rear))

# Display all results
print("_________________________________________________________")
print("_____________________ALL RESULTS_________________________")
print("_________________________________________________________")
print("Initial Guess for Main-Front Transformation: \n {}".format(initial_guess_main_front))
print("Optimal solution between main and front reference frames: \n {}".format(Xopt_main_front))
print("_________________________________________________________")
print("Initial Guess for Main-Rear Transformation: \n {}".format(initial_guess_main_rear))
print("Optimal solution between main and rear reference frames: \n {}".format(Xopt_main_rear))
print("_________________________________________________________")
print("Initial Guess for Front-Rear Transformation: \n {}".format(initial_guess_front_rear))
print("Optimal solution between front and rear reference frames: \n {}".format(Xopt_front_rear))
print("_________________________________________________________")

# Plot odometry of all three lidars
plt.figure()
plt.plot(main_odometry.x, color="r", label="Odometry, Main")
plt.plot(front_odometry.x, color="g", label="Odometry, Front")
plt.plot(rear_odometry.x, color="b", label="Odometry, Rear")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Odometric Position")
plt.title("Odometry Over Time of Main, Front, and Rear Lidar Sensors")
plt.savefig("all_odometry.png")
plt.show()
plt.clf()

# individual odometry plotting function
def plot(odom_df, lidar_type=None, color="b"):
    plt.plot(odom_df.x, color=color)
    plt.xlabel("Time")
    plt.ylabel("Odometric Position")
    plt.title("Odometry Over Time of {} Lidar Sensor".format(lidar_type))
    plt.savefig("odometry_{}.png".format(lidar_type))
    plt.show()
    plt.clf()

# Now create plots for other types of odometry
plot(main_odometry, lidar_type="Main", color="r")
plot(front_odometry, lidar_type="Front", color="g")
plot(rear_odometry, lidar_type="Rear", color="b")


