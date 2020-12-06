#!/usr/bin/python3

import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R

import autograd.numpy as np

from pymanopt.manifolds import Euclidean, Rotations, Product
from pymanopt import Problem
from pymanopt.solvers import TrustRegions
from custom_solver import CustomSteepestDescent

import relative_pose_processing
from extract_pose_yaml import load_transforms


# Specify odometry CSV file paths
MAIN_ODOM_CSV = 'main_odometry_clean.csv'
FRONT_ODOM_CSV = 'front_odometry_clean.csv'
REAR_ODOM_CSV = 'rear_odometry_clean.csv'

# Specify path for relative transformations between poses
PKL_POSES_PATH = 'relative_transforms.pkl'

# Path for results from manifold optimization
ANALYSIS_RESULTS_PATH = 'analysis_results'

# Create results directory
if not os.path.exists(ANALYSIS_RESULTS_PATH):
    os.mkdir(ANALYSIS_RESULTS_PATH)

def compute_weights(odom_df, type="main"):
    """Computes the covariance of the rotation matrix and the standard
    deviation of translation matrices to weight the samples in the optimization
    objective.

    Parameters:
        R (np.array): 3x3 rotation matrix extra
    """
    # Compute translations and compute covariance over all of them
    t = odom_df[["x", "y", "z"]].values  # Translation - with shape (N, 3)
    cov_t = np.cov(t.T)

    # Extract quaternions from odometric data frame
    Q = odom_df[["qx", "qy", "qz", "qw"]].values

    # Convert quaternions to euler angles
    E = np.zeros((Q.shape[0], 3))  # RPY angles array
    for i, q in enumerate(Q):  # Iterate over each quaternion
        E[i] = R.from_quat(q).as_euler('xyz', degrees=False)

    # From here, we need to extract rotations about x,y,z to compute covariance
    cov_E = np.cov(E.T)

    return cov_t, cov_E

def parse_icp_cov(odom_df, type="main"):
    """Function to parse the ICP covariance matrix for the given input df.  Saves
    a dictionary containing ICP covariance matrices to a pkl file.

    Paramaters:
        odom_df (pd.DataFrame):  DataFrame containing the (cleaned) odometric
            data for one of the lidars.

    Returns:
        cov_by_frame (dict):  Dictionary where each key is a frame number, and
            each value is a corresponding ICP 6x6 matrix for that specific frame.
    """
    # Load odometry
    cov_columns = odom_df.filter(regex=("cov_pose_*"))

    # Iteratively add ICP frame-by-frame
    cov_by_frame = {i: 0 for i in range(cov_columns.shape[0])}
    for i, row in enumerate(cov_columns.iterrows()):
        cov_by_frame[i] = np.array(row[1:]).reshape((6, 6))  # Reshape into 6 x 6 covariance matrix

    # Save ICP matrices to pkl files
    fname = "icp_cov_" + type + ".pkl"
    with open(fname, "wb") as pkl:
        pickle.dump(cov_by_frame, pkl)
        pkl.close()

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
main_icp = parse_icp_cov(main_odometry, type="main")
front_icp = parse_icp_cov(front_odometry, type="front")
rear_icp = parse_icp_cov(rear_odometry, type="rear")

# Compute the covariance of translation and rotation (the latter uses Euler angles)
cov_t_main, cov_R_main = compute_weights(main_odometry, type="main")
cov_t_front, cov_R_front = compute_weights(front_odometry, type="front")
cov_t_rear, cov_R_rear = compute_weights(rear_odometry, type="rear")

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


######################## MAIN FRONT CALIBRATION ################################
# Carry out optimization for main-front homogeneous transformations
problem_main_front = Problem(manifold=manifold, cost=cost_main_front)  # (2a) Compute the optimization between main and front
solver_main_front = CustomSteepestDescent()  # (3) Instantiate a Pymanopt solver
Xopt_main_front = solver_main_front.solve(problem_main_front, x=X0_main_front)
print("Initial Guess for Main-Front Transformation: \n {}".format(initial_guess_main_front))
print("Optimal solution between main and front reference frames: \n {}".format(Xopt_main_front))

# Take intermediate values for plotting
estimates_x_main_front = solver_main_front.estimates
errors_main_front = solver_main_front.errors
iters_main_front = solver_main_front.iterations

# Metrics dictionary
estimates_dict_main_front = {i: T for i, T in zip(iters_main_front, estimates_x_main_front)}
error_dict_main_front = {i: e for i, e in zip(iters_main_front, errors_main_front)}

# Save intermediate results to a pkl file
estimates_fname_main_front = os.path.join(ANALYSIS_RESULTS_PATH,
                                          "estimates_main_front.pkl")
error_fname_main_front = os.path.join(ANALYSIS_RESULTS_PATH,
                                          "error_main_front.pkl")

# Save estimates to pickle file
with open(estimates_fname_main_front, "wb") as pkl_estimates:
    pickle.dump(estimates_dict_main_front, pkl_estimates)
    pkl_estimates.close()

# Save error to pickle file
with open(error_fname_main_front, "wb") as pkl_error:
    pickle.dump(error_dict_main_front, pkl_error)
    pkl_error.close()
################################################################################

######################## MAIN REAR CALIBRATION #################################
# Carry out optimization for main-rear homogeneous transformations
problem_main_rear = Problem(manifold=manifold, cost=cost_main_rear)  # (2a) Compute the optimization between main and front
solver_main_rear = CustomSteepestDescent()  # (3) Instantiate a Pymanopt solver
Xopt_main_rear = solver_main_rear.solve(problem_main_rear, x=X0_main_rear)
print("Initial Guess for Main-Rear Transformation: \n {}".format(initial_guess_main_rear))
print("Optimal solution between main and rear reference frames: \n {}".format(Xopt_main_rear))

# Take intermediate values for plotting
estimates_x_main_rear = solver_main_rear.estimates
errors_main_rear = solver_main_rear.errors
iters_main_rear = solver_main_rear.iterations

# Metrics dictionary
estimates_dict_main_rear = {i: T for i, T in zip(iters_main_rear, estimates_x_main_rear)}
error_dict_main_rear = {i: e for i, e in zip(iters_main_rear, errors_main_rear)}

# Save intermediate results to a pkl file
estimates_fname_main_rear = os.path.join(ANALYSIS_RESULTS_PATH,
                                          "estimates_main_rear.pkl")
error_fname_main_rear = os.path.join(ANALYSIS_RESULTS_PATH,
                                      "error_main_rear.pkl")

# Save estimates to pickle file
with open(estimates_fname_main_rear, "wb") as pkl_estimates:
    pickle.dump(estimates_dict_main_rear, pkl_estimates)
    pkl_estimates.close()

# Save error to pickle file
with open(error_fname_main_rear, "wb") as pkl_error:
    pickle.dump(error_dict_main_rear, pkl_error)
    pkl_error.close()
################################################################################

######################## FRONT REAR CALIBRATION ################################
# Carry out optimization for front-rear homogeneous transformations
problem_front_rear = Problem(manifold=manifold, cost=cost_front_rear)  # (2a) Compute the optimization between main and front
solver_front_rear = CustomSteepestDescent()  # (3) Instantiate a Pymanopt solver
Xopt_front_rear = solver_front_rear.solve(problem_front_rear, x=X0_front_rear)
print("Initial Guess for Front-Rear Transformation: \n {}".format(initial_guess_front_rear))
print("Optimal solution between front and rear reference frames: \n {}".format(Xopt_front_rear))

# Take intermediate values for plotting
estimates_x_front_rear = solver_front_rear.estimates
errors_front_rear = solver_front_rear.errors
iters_front_rear = solver_front_rear.iterations

# Metrics dictionary
estimates_dict_front_rear = {i: T for i, T in zip(iters_front_rear, estimates_x_front_rear)}
error_dict_front_rear = {i: e for i, e in zip(iters_front_rear, errors_front_rear)}

# Save intermediate results to a pkl file
estimates_fname_front_rear = os.path.join(ANALYSIS_RESULTS_PATH,
                                          "estimates_front_rear.pkl")
error_fname_front_rear = os.path.join(ANALYSIS_RESULTS_PATH,
                                      "error_front_rear.pkl")

# Save estimates to pickle file
with open(estimates_fname_front_rear, "wb") as pkl_estimates:
    pickle.dump(estimates_dict_front_rear, pkl_estimates)
    pkl_estimates.close()

# Save error to pickle file
with open(error_fname_front_rear, "wb") as pkl_error:
    pickle.dump(error_dict_front_rear, pkl_error)
    pkl_error.close()
################################################################################


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
def plot_odometry(odom_df, lidar_type=None, color="b"):
    """Plotting function for odometry."""
    plt.plot(odom_df.x, color=color)
    plt.xlabel("Time")
    plt.ylabel("Odometric Position")
    plt.title("Odometry Over Time of {} Lidar Sensor".format(lidar_type))
    plt.savefig("odometry_{}.png".format(lidar_type))
    plt.show()
    plt.clf()

def plot_error(err, lidar_type=None, color="b"):
    """Plotting function for error as a function of optimization iteration."""
    plt.plot(err, color=color)
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.title("Error vs. Optimization Iteration for Lidar {}".format(lidar_type))
    plt.savefig("odometry_error_{}.png".format(lidar_type))
    plt.show()
    plt.clf()


# Now create plots for other types of odometry
plot_odometry(main_odometry, lidar_type="Main", color="r")
plot_odometry(front_odometry, lidar_type="Front", color="g")
plot_odometry(rear_odometry, lidar_type="Rear", color="b")


# Now plot errors as a function of iterations
plot_error(errors_main_front, lidar_type="Main", color="r")
plot_error(errors_main_rear, lidar_type="Front", color="g")
plot_error(errors_front_rear, lidar_type="Rear", color="b")
