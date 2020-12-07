#!/usr/bin/python3

"""Functions and function calls for computing and estimating the relative
transformations between our lidar frames."""

# Native Python imports
import pickle
import os

# For autodiff with manifold optimization
import autograd.numpy as np

# Manifold optimization
from pymanopt.manifolds import Euclidean, Rotations, Product
from pymanopt import Problem
from custom_solver import CustomSteepestDescent

# Custom import
import relative_pose_processing
from extract_pose_yaml import load_transforms, construct_pose
from analysis_utils import cost, parse_icp_cov, plot_all_odom, plot_covs, \
    plot_error, compute_weights_euler, extract_variance, compute_rmse_weighted, \
    compute_rmse_unweighted, display_and_save_rmse

# For rejecting samples if the ICP covariance is too high
REJECT_THR = 100

# Whether to weight our estimate
WEIGHTED = True

# Specify odometry CSV file paths
MAIN_ODOM_CSV = 'main_odometry_clean.csv'
FRONT_ODOM_CSV = 'front_odometry_clean.csv'
REAR_ODOM_CSV = 'rear_odometry_clean.csv'

# Specify path for relative transformations between poses
PKL_POSES_PATH = 'relative_transforms.pkl'

# Path for results from manifold optimization
ANALYSIS_RESULTS_PATH = 'analysis_results_weighted_{}'.format(WEIGHTED)
FINAL_ESTIMATES_PATH = 'final_estimates_weighted_{}'.format(WEIGHTED)
ODOMETRY_PLOTS_PATH = 'odometry_plots_weighted_{}'.format(WEIGHTED)

# Create results directories
if not os.path.exists(ANALYSIS_RESULTS_PATH):  # Where errors and all estimates are stored
    os.mkdir(ANALYSIS_RESULTS_PATH)
    
if not os.path.exists(FINAL_ESTIMATES_PATH):  # Where estimates are stored
    os.mkdir(FINAL_ESTIMATES_PATH)

if not os.path.exists(ODOMETRY_PLOTS_PATH):  # Where odometry plots are saved
    os.mkdir(ODOMETRY_PLOTS_PATH)

def make_all_plots():
    """Function to create all plots from this analysis."""
    # Now create plots for other types of odometry
    plot_odometry(main_odometry, lidar_type="Main", color="r")
    plot_odometry(front_odometry, lidar_type="Front", color="g")
    plot_odometry(rear_odometry, lidar_type="Rear", color="b")

    # Now plot errors as a function of iterations
    plot_error(errors_main_front, lidar_type="Main", color="r")
    plot_error(errors_main_rear, lidar_type="Front", color="g")
    plot_error(errors_front_rear, lidar_type="Rear", color="b")

    # Create the plots
    plot_covs(main_rot_cov, main_trans_cov, lidar_type="Main", clip=True)
    plot_covs(front_rot_cov, front_trans_cov, lidar_type="Front", clip=True)
    plot_covs(rear_rot_cov, rear_trans_cov, lidar_type="Rear", clip=True)

def main():
    # Extract and process the CSVs
    main_odometry = relative_pose_processing.process_df(MAIN_ODOM_CSV)
    front_odometry = relative_pose_processing.process_df(FRONT_ODOM_CSV)
    rear_odometry = relative_pose_processing.process_df(REAR_ODOM_CSV)

    # Process poses
    (main_aligned, front_aligned, rear_aligned) = relative_pose_processing.align_df([main_odometry, front_odometry, rear_odometry])

    # Get ICP covariance matrices
    # Main lidar odometry
    main_icp, main_trans_cov, main_trans_cov_max, \
        main_trans_cov_avg, main_rot_cov, main_rot_cov_max, \
        main_rot_cov_avg, main_reject = parse_icp_cov(main_odometry, type="main",
                                                      reject_thr=REJECT_THR)

    # Front lidar odometry
    front_icp, front_trans_cov, front_trans_cov_max, \
        front_trans_cov_avg, front_rot_cov, front_rot_cov_max, \
        front_rot_cov_avg, front_reject = parse_icp_cov(front_odometry, type="front",
                                                        reject_thr=REJECT_THR)

    # Rear lidar odometry
    rear_icp, rear_trans_cov, rear_trans_cov_max, \
        rear_trans_cov_avg, rear_rot_cov, rear_rot_cov_max, \
        rear_rot_cov_avg, rear_reject = parse_icp_cov(rear_odometry, type="rear",
                                                      reject_thr=REJECT_THR)

    # Compute the covariance of translation and rotation (the latter uses Euler angles)
    cov_t_main, cov_R_main = compute_weights_euler(main_odometry, type="main")
    cov_t_front, cov_R_front = compute_weights_euler(front_odometry, type="front")
    cov_t_rear, cov_R_rear = compute_weights_euler(rear_odometry, type="rear")

    # Extract a single scalar using the average value from rotation and translation
    var_t_main = extract_variance(cov_t_main, mode="avg")
    var_R_main = extract_variance(cov_R_main, mode="avg")
    var_t_front = extract_variance(cov_t_front, mode="avg")
    var_R_front = extract_variance(cov_R_front, mode="avg")
    var_t_rear = extract_variance(cov_t_main, mode="avg")
    var_R_rear = extract_variance(cov_R_rear, mode="avg")

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
    direct_initial_guess_front_rear = transforms_dict["direct_front_rear"]  # Transform directly computed


    # Print out all the initial estimates as poses
    print("INITIAL GUESS MAIN FRONT: \n {} \n".format(initial_guess_main_front))
    print("INITIAL GUESS MAIN REAR: \n {} \n".format(initial_guess_main_rear))
    print("INITIAL GUESS FRONT REAR: \n {} \n".format(initial_guess_front_rear))
    print("INITIAL GUESS DIRECT FRONT REAR: \n {} \n".format(direct_initial_guess_front_rear))
    
    
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
    ### PARAMETERS ###
    A = np.array(front_rel_poses)  # First set of poses
    B = np.array(main_rel_poses)  # Second set of poses
    N = min(A.shape[0], B.shape[0])
    r = np.logical_or(np.array(main_reject[:N]), np.array(front_reject[:N]))  # If either has high variance, reject the sample
    omega = np.mean([var_R_main, var_R_front])  # Take average across different odometries
    rho = np.mean([var_t_main, var_t_front])  # Take average across different odometries
    ### PARAMETERS ###

    cost_main_front = lambda x: cost(x, A, B, r, rho, omega, WEIGHTED)
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
    
    # Calculate difference between initial guess and final
    XOpt_T_main_front = construct_pose(Xopt_main_front[0], Xopt_main_front[1].reshape((3, 1)))
    print("DIFFERENCE IN MATRICES: \n {}".format(np.subtract(XOpt_T_main_front, initial_guess_main_front)))

    # Compute the weighted and unweighted RMSE
    rmse_init_weighted, rmse_final_weighted, rmse_init_R_weighted, \
    rmse_init_t_weighted, rmse_final_R_weighted, \
    rmse_final_t_weighted = compute_rmse_weighted(initial_guess_main_front,
                                                  XOpt_T_main_front, A, B, rho,
                                                  omega)
    rmse_init_unweighted, rmse_final_unweighted, rmse_init_R_unweighted, \
    rmse_init_t_unweighted, rmse_final_R_unweighted, \
    rmse_final_t_unweighted = compute_rmse_unweighted(initial_guess_main_front,
                                                      XOpt_T_main_front, A, B)
    rmses = [rmse_init_unweighted, rmse_final_unweighted, rmse_init_weighted, rmse_final_weighted,
             rmse_init_R_unweighted, rmse_init_t_unweighted,
             rmse_final_R_unweighted, rmse_final_t_unweighted,
             rmse_init_R_weighted, rmse_init_t_weighted,
             rmse_final_R_weighted, rmse_final_t_weighted]

    # Display and save RMSEs
    outpath = os.path.join(ANALYSIS_RESULTS_PATH, "main_front_rmse.txt")
    display_and_save_rmse(rmses, outpath)
    
    # Save final estimates
    final_estimate_outpath = os.path.join(FINAL_ESTIMATES_PATH,
                                          "main_front_final.txt")
    np.savetxt(final_estimate_outpath, XOpt_T_main_front)
    ################################################################################


    ######################## MAIN REAR CALIBRATION #################################
    ### PARAMETERS ###
    A = np.array(rear_rel_poses)  # First set of poses
    B = np.array(main_rel_poses)  # Second set of poses
    N = min(A.shape[0], B.shape[0])
    r = np.logical_or(np.array(main_reject[:N]), np.array(rear_reject[:N]))  # If either has high variance, reject the sample
    omega = np.mean([var_R_main, var_R_rear])  # Take average across different odometries
    rho = np.mean([var_t_main, var_t_rear])  # Take average across different odometries
    ### PARAMETERS ###

    cost_main_rear = lambda x: cost(x, A, B, r, rho, omega, WEIGHTED)
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
    
    # Calculate difference between initial guess and final
    XOpt_T_main_rear = construct_pose(Xopt_main_rear[0], Xopt_main_rear[1].reshape((3, 1)))
    print("DIFFERENCE IN MATRICES: \n {}".format(np.subtract(XOpt_T_main_rear, initial_guess_main_rear)))

    # Compute the weighted and unweighted RMSE
    rmse_init_weighted, rmse_final_weighted, rmse_init_R_weighted, \
           rmse_init_t_weighted, rmse_final_R_weighted, \
           rmse_final_t_weighted = compute_rmse_weighted(initial_guess_main_rear, XOpt_T_main_rear, A, B, rho, omega)
    rmse_init_unweighted, rmse_final_unweighted, rmse_init_R_unweighted, \
           rmse_init_t_unweighted, rmse_final_R_unweighted, \
           rmse_final_t_unweighted = compute_rmse_unweighted(initial_guess_main_rear, XOpt_T_main_rear, A, B)
    rmses = [rmse_init_unweighted, rmse_final_unweighted, rmse_init_weighted, rmse_final_weighted,
             rmse_init_R_unweighted, rmse_init_t_unweighted,
             rmse_final_R_unweighted, rmse_final_t_unweighted,
             rmse_init_R_weighted, rmse_init_t_weighted,
             rmse_final_R_weighted, rmse_final_t_weighted]

    # Display and save RMSEs
    outpath = os.path.join(ANALYSIS_RESULTS_PATH, "main_rear_rmse.txt")
    display_and_save_rmse(rmses, outpath)

    # Save final estimates
    final_estimate_outpath = os.path.join(FINAL_ESTIMATES_PATH,
                                          "main_rear_final.txt")
    np.savetxt(final_estimate_outpath, XOpt_T_main_rear)
    ################################################################################



    ######################## FRONT REAR CALIBRATION ################################
    ### PARAMETERS ###
    A = np.array(rear_rel_poses)  # First set of poses
    B = np.array(front_rel_poses)  # Second set of poses
    N = min(A.shape[0], B.shape[0])
    r = np.logical_or(np.array(front_reject[:N]), np.array(rear_reject[:N]))  # If either has high variance, reject the sample
    omega = np.mean([var_R_front, var_R_rear])  # Take average across different odometries
    rho = np.mean([var_t_front, var_t_rear])  # Take average across different odometries
    ### PARAMETERS ###

    cost_front_rear = lambda x: cost(x, A, B, r, rho, omega, WEIGHTED)
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
    
    # Calculate difference between initial guess and final
    XOpt_T_front_rear = construct_pose(Xopt_front_rear[0], Xopt_front_rear[1].reshape((3, 1)))
    print("DIFFERENCE IN MATRICES: \n {}".format(np.subtract(XOpt_T_front_rear, initial_guess_front_rear)))

    # Compute the weighted and unweighted RMSE
    rmse_init_weighted, rmse_final_weighted, rmse_init_R_weighted, \
    rmse_init_t_weighted, rmse_final_R_weighted, \
    rmse_final_t_weighted = compute_rmse_weighted(initial_guess_front_rear,
                                                  XOpt_T_front_rear, A, B, rho,
                                                  omega)
    rmse_init_unweighted, rmse_final_unweighted, rmse_init_R_unweighted, \
    rmse_init_t_unweighted, rmse_final_R_unweighted, \
    rmse_final_t_unweighted = compute_rmse_unweighted(initial_guess_front_rear,
                                                      XOpt_T_front_rear, A, B)
    rmses = [rmse_init_unweighted, rmse_final_unweighted, rmse_init_weighted,
             rmse_final_weighted,
             rmse_init_R_unweighted, rmse_init_t_unweighted,
             rmse_final_R_unweighted, rmse_final_t_unweighted,
             rmse_init_R_weighted, rmse_init_t_weighted,
             rmse_final_R_weighted, rmse_final_t_weighted]

    # Display and save RMSEs
    outpath = os.path.join(ANALYSIS_RESULTS_PATH, "front_rear_rmse.txt")
    display_and_save_rmse(rmses, outpath)

    # Save final estimates
    final_estimate_outpath = os.path.join(FINAL_ESTIMATES_PATH,
                                          "front_rear_final.txt")
    np.savetxt(final_estimate_outpath, XOpt_T_front_rear)
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

if __name__ == '__main__':
    main()