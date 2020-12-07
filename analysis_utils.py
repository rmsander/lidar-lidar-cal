#!/usr/bin/python3

"""Utility functions for plotting odometry and ICP covariance."""
import os
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
import autograd.numpy as np

# For rejecting samples if the ICP covariance is too high
REJECT_THR = 10

def _plot_icp_cov(cov, lidar_type=None, comp="Rotation",
                  coord="x", color="b", results_dir="icp_plots"):
    """Helper function for plotting ICP functions."""
    # Set out path
    if not os.path.exists(os.path.join(results_dir, lidar_type)):
        os.mkdir(os.path.join(results_dir, lidar_type))
    out_path = os.path.join(results_dir, lidar_type,
                            "cov_lidar-{}_comp-{}_coord-{}.png".format(lidar_type, comp, coord))

    # Create plot
    plt.plot(cov, color=color)
    plt.xlabel("Timesteps (0.1 seconds)")
    plt.ylabel("Covariance, {}, {}-component".format(comp, coord))
    plt.title("Covariance, {}, {}-component vs. Time for {} Lidar".format(comp, coord, lidar_type))
    plt.savefig(out_path)
    plt.clf()

def plot_covs(cov_R, cov_t, lidar_type="Front", clip=False, clip_val=1):
    """Function for plotting all ICP covariances of a matrix."""
    cov_R_as_list = [cov_R[i] for i in range(len(list(cov_R.keys())))]
    cov_t_as_list = [cov_t[i] for i in range(len(list(cov_t.keys())))]

    # Get covariances for rotation
    if clip:
        cov_rx = [c[0] if c[0] < clip_val else clip_val for c in cov_R_as_list]
        cov_ry = [c[1] if c[1] < clip_val else clip_val for c in cov_R_as_list]
        cov_rz = [c[2] if c[2] < clip_val else clip_val for c in cov_R_as_list]

        # Get covariances for translation
        cov_tx = [c[0] if c[0] < clip_val else clip_val for c in cov_t_as_list]
        cov_ty = [c[1] if c[1] < clip_val else clip_val for c in cov_t_as_list]
        cov_tz = [c[2] if c[2] < clip_val else clip_val for c in cov_t_as_list]

    else:
        cov_rx = [c[0] for c in cov_R_as_list]
        cov_ry = [c[1] for c in cov_R_as_list]
        cov_rz = [c[2] for c in cov_R_as_list]

        # Get covariances for translation
        cov_tx = [c[0] for c in cov_t_as_list]
        cov_ty = [c[1] for c in cov_t_as_list]
        cov_tz = [c[2] for c in cov_t_as_list]

    # Concatenate variables for repeated plotted
    covs = [cov_rx, cov_ry, cov_rz, cov_tx, cov_ty, cov_tz]
    comps = ["Rotation", "Rotation", "Rotation",
            "Translation", "Translation", "Translation"]
    coords = ["x", "y", "z", "x", "y", "z"]
    colors = ["b", "g", "y", "c", "r", "m"]

    # Now create plots
    if clip:
        results_dir = "icp_plots_clipped"
    else:
        results_dir = "icp_plots"

    # Make sure results directory exists
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Iteratively create plots for each sensor and ICP cov component
    for cov, comp, coord, color in zip(covs, comps, coords, colors):
        _plot_icp_cov(cov, lidar_type=lidar_type, comp=comp,
                      coord=coord, color=color, results_dir=results_dir)


# Plot odometry of all three lidars
def plot_all_odom(main_odometry, front_odometry, rear_odometry):
    """Function for plotting odometry of all three lidar sensors."""
    plt.figure()
    plt.plot(main_odometry.x, color="r", label="Odometry, Main")
    plt.plot(front_odometry.x, color="g", label="Odometry, Front")
    plt.plot(rear_odometry.x, color="b", label="Odometry, Rear")
    plt.legend()
    plt.xlabel("Timestep (0.1 seconds)")
    plt.ylabel("Odometric Position")
    plt.title("Odometry Over Time of Main, Front, and Rear Lidar Sensors")
    plt.savefig("all_odometry.png")
    plt.show()
    plt.clf()

# individual odometry plotting function
def plot_odometry(odom_df, lidar_type=None, color="b"):
    """Plotting function for odometry."""
    plt.plot(odom_df.x, color=color)
    plt.xlabel("Timestep (0.1 seconds)")
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

    # Get all diagonals, max, and average of translation
    trans_cov = {i: 0 for i in range(cov_columns.shape[0])}
    trans_cov_max = {i: 0 for i in range(cov_columns.shape[0])}
    trans_cov_avg = {i: 0 for i in range(cov_columns.shape[0])}

    # Get all diagonals, max, and average of rotation
    rot_cov = {i: 0 for i in range(cov_columns.shape[0])}
    rot_cov_max = {i: 0 for i in range(cov_columns.shape[0])}
    rot_cov_avg = {i: 0 for i in range(cov_columns.shape[0])}

    # Used for determining if samples should be rejected
    reject = [False for i in range(cov_columns.shape[0])]
    for i in range(len(reject)):
        if rot_cov_max[i] > REJECT_THR:
            reject[i] = True

    for i in range(len(list(cov_by_frame.keys()))):

        # ICP covariance matrix
        icp_cov = cov_by_frame[i]

        # Translation - upper 3 x 3 of ICP covariance
        trans_cov[i] = [icp_cov[0, 0], icp_cov[1, 1], icp_cov[2, 2]]
        trans_cov_max[i] = max(trans_cov[i])
        trans_cov_avg[i] = np.mean(trans_cov[i])

        # Rotation - lower 3 x 3 of ICP covariance
        rot_cov[i] = [icp_cov[3, 3], icp_cov[4, 4], icp_cov[5, 5]]
        rot_cov_max[i] = max(rot_cov[i])
        rot_cov_avg[i] = np.mean(rot_cov[i])


    return cov_by_frame, trans_cov, trans_cov_max, \
           trans_cov_avg, rot_cov, rot_cov_max, rot_cov_avg, reject


def cost(X, A, B, r, rho, omega):
    """Function to compute the cost between two sets of poses given by the
    lists A and B.  This cost function is passed in as a parameter to the
    manifold SE(3) manifold optimization.

    This cost function is computed between the main and front poses.

    Parameters:
        X (tuple): A tuple containing a 3x3 rotation matrix estimate given by the
            R, and a 3 x1 translation vector t.  The manifold optimization is
            carried out in SE(3).
        A (list or np.array):  Array of 4 x 4 poses computed from the odometry data.
        B (list or np.array):  Array of 4 x 4 poses computed from the odometry data.
        r (list):  A list of booleans corresponding to whether we take the current
            sample, according to whether its ICP covariance is high or low.
        rho (float):  Float corresponding to the variance weighting for translation.
        omega (float):  Float corresponding to the variance weighting for translation.

    Returns:
        cost (float):  The total cost computed over the sets of poses.
    """
    # Initialize cost
    cost = 0

    # Estimates for rotation (R) and translation (t)
    R, t = X

    # Construct 4 x 4 pose estimate manually
    R_t = np.hstack((R, t.reshape((3, 1))))  # 3 x 4 upper block
    T4 = np.array([0, 0, 0, 1])  # Use 1 in last entry for homogeneous coordinates
    Tab = np.vstack((R_t, T4))  # Merge 3 x 4 with 1 x 4 to get pose

    # Sum the cost over all poses
    for ix in range(len(A)):
        if not r[ix]:  # If we don't reject the sample, compute cost
            M = B[ix] @ Tab @ np.linalg.inv(A[ix]) - Tab
            O = np.array([[omega, 0, 0, 0],
                          [0, omega, 0, 0],
                          [0, 0, omega, 0],
                          [0, 0, 0, rho]])
            cost += np.trace(M @ O @ M.T)

    return cost

def compute_rmse_unweighted(X_init, X_final, A, B):
    """Computes the unweighted RMSE over the entire trajectory.  Used as a benchmark to
    determine if our framework improves the estimate of these relative pose
    transformations.

    Parameters:
        X_init (np.array): 4 x 4 array corresponding to our initial pose
            estimate.  This is the relative pose transformation extracted and
            computed from the husky yaml file.
        X_init (np.array): 4 x 4 array corresponding to our initial pose
            estimate.  This is the relative pose transformation extracted and
            computed from the husky yaml file.
        A (list or np.array):  Array of 4 x 4 poses computed from the odometry data.
        B (list or np.array):  Array of 4 x 4 poses computed from the odometry data.

    Returns:
        rmse_initial (float): Root Mean Squared Error from the initial estimate.
        rmse_final (float): Root Mean Squared Error from the final estimate.
    """
    # Initialize mse for initial and final estimates
    mse_init = 0
    mse_final = 0

    # Check to make sure that A and B are autograd arrays
    A = np.array(A)
    B = np.array(B)

    # Get number of samples
    N = len(A)

    # Sum the cost over all poses
    for ix in range(N):

        # Compute element-wise cost term for initial estimate
        M_init = B[ix] @ X_init @ np.linalg.inv(A[ix]) - X_init
        mse_init += np.trace(M_init @ M_init.T)

        # Compute element-wise cost term for initial estimate
        M_final = B[ix] @ X_final @ np.linalg.inv(A[ix]) - X_final
        mse_final += np.trace(M_final @ M_final.T)

    # Finally, take the square root to get rmse
    rmse_init = np.sqrt(mse_init / N)
    rmse_final = np.sqrt(mse_final / N)

    return rmse_init, rmse_final

def compute_rmse_weighted(X_init, X_final, A, B, rho, omega):
    """Computes the weighted RMSE over the entire trajectory.  Used as a benchmark to
    determine if our framework improves the estimate of these relative pose
    transformations.

    Parameters:
        X_init (np.array): 4 x 4 array corresponding to our initial pose
            estimate.  This is the relative pose transformation extracted and
            computed from the husky yaml file.
        X_init (np.array): 4 x 4 array corresponding to our initial pose
            estimate.  This is the relative pose transformation extracted and
            computed from the husky yaml file.
        A (list or np.array):  Array of 4 x 4 poses computed from the odometry data.
        B (list or np.array):  Array of 4 x 4 poses computed from the odometry data.

    Returns:
        rmse_initial (float): Root Mean Squared Error from the initial estimate.
        rmse_final (float): Root Mean Squared Error from the final estimate.
    """
    # Initialize mse for initial and final estimates
    mse_init = 0
    mse_final = 0

    # Check to make sure that A and B are autograd arrays
    A = np.array(A)
    B = np.array(B)

    # Get number of samples
    N = len(A)

    # Sum the cost over all poses
    for ix in range(N):

        # Define the weighting matrix
        O = np.array([[omega, 0, 0, 0],
                      [0, omega, 0, 0],
                      [0, 0, omega, 0],
                      [0, 0, 0, rho]])

        # Compute element-wise cost term for initial estimate
        M_init = B[ix] @ X_init @ np.linalg.inv(A[ix]) - X_init
        mse_init += np.trace(M_init @ O @ M_init.T)

        # Compute element-wise cost term for initial estimate
        M_final = B[ix] @ X_final @ np.linalg.inv(A[ix]) - X_final
        mse_final += np.trace(M_final @ O @ M_final.T)

    # Finally, take the square root to get rmse
    rmse_init = np.sqrt(mse_init / N)
    rmse_final = np.sqrt(mse_final / N)

    return rmse_init, rmse_final

def compute_weights_euler(odom_df, type="main"):
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

def display_and_save_rmse(rmses, outpath):
    """Function to display different RMSE values from weighted and unweighted
    estimates and save them to an output file.

    Parameters:
        rmses (list): A list with the order of rmses given as:
            [rmse_init_unweighted, rmse_final_unweighted,
             rmse_init_weighted, rmse_final_weighted]
        outpath (str): A path (absolute or relative) where the rmse values
            will be saved.
    """
    # Get different rmse values
    rmse_init_unweighted, rmse_final_unweighted, \
        rmse_init_weighted, rmse_final_weighted = rmses

    # Display RMSE values
    print("RMSE: \n"
          "  INITIAL UNWEIGHTED: {} \n"
          "  FINAL UNWEIGHTED: {} \n"
          "  INITIAL WEIGHTED: {} \n"
          "  FINAL WEIGHTED: {} \n".format(rmse_init_unweighted,
                                           rmse_final_unweighted,
                                           rmse_init_weighted,
                                           rmse_final_weighted)
          )
    # Save RMSE values to a text file
    with open(outpath, "w") as rmse_txt_file:
        # Write files
        rmse_txt_file.write(
            "INITIAL UNWEIGHTED: {} \n".format(rmse_init_unweighted))
        rmse_txt_file.write(
            "FINAL UNWEIGHTED: {} \n".format(rmse_final_unweighted))
        rmse_txt_file.write(
            "INITIAL WEIGHTED: {} \n".format(rmse_init_weighted))
        rmse_txt_file.write("FINAL WEIGHTED: {} \n".format(rmse_final_weighted))
        rmse_txt_file.close()

def extract_variance(cov, mode="avg"):
    """Function to extract the variance for weighting our relative pose
    estimate over a rigid-body transformation.

    Parameters:
        cov (np.array): A 3x3 matrix corresponding to a covariance matrix,
            for either the covariance of a set of translations, or over
            rotations given by a vector of Euler angles.

    Returns:
        var (float):  Float value corresponding to the variance used for
            weighting our relative pose estimate.
    """
    if mode == "avg":  # Average diagonal elements
        return np.trace(cov) / cov.shape[0]

    elif mode == "max":  # Take maximum over diagonal elements
        return max([cov[0, 0], cov[1, 1], cov[2, 2]])

    else:
        print("Invalid mode specified.  Please choose from {avg, max}.")