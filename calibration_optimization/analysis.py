#!/usr/bin/python3

"""Functions and function calls for computing and estimating the relative
transformations between our lidar frames.  A few functions this implements:

1. Nonlinear unweighted manifold optimization over SE(3) to compute an optimal
pose.  This can be performed over the entire dataset, as well as in a K-fold
cross-validation fashion.

2. Nonlinear weighted manifold optimization over SE(3) to compute an optimal
pose.  This can be performed over the entire dataset, as well as in a K-fold
cross-validation fashion.  Weights are computed according to the maximum
variance of translation and rotation odometry measurements.  Rejection variables
are determined by the maximum variance of the ICP measurements.
"""

# Native Python imports
import pickle
import os
import argparse

# For auto-differentiation with manifold optimization
import autograd.numpy as np

# Manifold optimization
from pymanopt.manifolds import Euclidean, Rotations, Product
from pymanopt import Problem
from custom_solver import CustomSteepestDescent

# Custom imports
import relative_pose_processing
from extract_pose_yaml import load_transforms, construct_pose
from analysis_utils import cost, parse_icp_cov, \
    plot_covs, plot_error, compute_weights_euler, extract_variance, \
    compute_rmse_weighted, compute_rmse_unweighted, display_and_save_rmse
from load_utils import check_dir

# Cross-validation
from sklearn.model_selection import KFold

def make_all_plots():
    """Function to create all plots from this calibration_optimization."""
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


def cross_validation(odom_1, aligned_1, odom_2, aligned_2, type_1, type_2, K=10):
    """Function to run cross-validation to run nonlinear optimization for optimal
    pose estimation and evaluation.  Performs cross-validation K times and splits
    the dataset into K (approximately) even splits, to be used for in-sample
    training and out-of-sample evaluation.

    This function estimates a relative transformation between two lidar frames
    using nonlinear optimization, and evaluates the robustness of this estimate
    through K-fold cross-validation performance of our framework.  Though this
    function does not return any values, it saves all results in the
    'results' relative path.

    Parameters:
        odom_1 (pd.DataFrame):  DataFrame corresponding to odometry data for the
            pose we wish to transform into the odom_2 frame of reference.  See
            data/main_odometry.csv for an example of the headers/columns/data
            types this function expects this DataFrame to have.

        aligned_1 (pd.DataFrame): DataFrame corresponding to aligned odometry
            data given the 3 sets of odometry data for the 3 lidar sensors.  This
            data corresponds to the odom_1 sensor frame.

        odom_2 (pd.DataFrame):  DataFrame corresponding to odometry data for the
            pose we wish to transform the odom_1 frame of reference into.  See
            data/main_odometry.csv for an example of the headers/columns/data
            types this function expects this DataFrame to have.

        aligned_2 (pd.DataFrame): DataFrame corresponding to aligned odometry
            data given the 3 sets of odometry data for the 3 lidar sensors.  This
            data corresponds to the odom_2 sensor frame.

        type_1 (str):  String denoting the lidar type.  Should be in the set
            {'main', 'front', 'rear'}.  This type corresponds to the data type
            for the odom_1 frame.

        type_2 (str):  String denoting the lidar type.  Should be in the set
            {'main', 'front', 'rear'}.  This type corresponds to the data type
            for the odom_2 frame.

        K (int):  The number of folds to be used for cross-validation.  Defaults
            to 10.
    """
    # Get ICP covariance matrices
    # Odom 1 lidar odometry
    odom1_icp, odom1_trans_cov, odom1_trans_cov_max, \
    odom1_trans_cov_avg, odom1_rot_cov, odom1_rot_cov_max, \
    odom1_rot_cov_avg, odom1_reject = parse_icp_cov(odom_1, type=type_1,
                                                  reject_thr=REJECT_THR)

    # Odom 2 lidar odometry
    odom2_icp, odom2_trans_cov, odom2_trans_cov_max, \
    odom2_trans_cov_avg, odom2_rot_cov, odom2_rot_cov_max, \
    odom2_rot_cov_avg, odom2_reject = parse_icp_cov(odom_2, type=type_2,
                                                    reject_thr=REJECT_THR)
    # Calculate relative poses
    (odom1_aligned, odom1_rel_poses) = relative_pose_processing.calc_rel_poses(aligned_1)
    (odom2_aligned, odom2_rel_poses) = relative_pose_processing.calc_rel_poses(aligned_2)

    # Compute weights for weighted estimate
    cov_t_odom1, cov_R_odom1 = compute_weights_euler(odom1_aligned)
    cov_t_odom2, cov_R_odom2 = compute_weights_euler(odom2_aligned)

    # Extract a single scalar using the average value from rotation and translation
    var_t_odom1 = extract_variance(cov_t_odom1, mode="max")
    var_R_odom1 = extract_variance(cov_R_odom1, mode="max")
    var_t_odom2 = extract_variance(cov_t_odom2, mode="max")
    var_R_odom2 = extract_variance(cov_R_odom2, mode="max")


    # Optimization (1) Instantiate a manifold
    translation_manifold = Euclidean(3)  # Translation vector
    so3 = Rotations(3)  # Rotation matrix
    manifold = Product((so3, translation_manifold))  # Instantiate manifold

    # Get initial guesses for our estimations
    if os.path.exists(PKL_POSES_PATH):  # Check to make sure path exists
        transforms_dict = load_transforms(PKL_POSES_PATH)  # Relative transforms

    # Map types to sensor names to access initial estimate relative transforms
    types2sensors = {"main": "velodyne",
                     "front": "front",
                     "rear": "rear"}

    # Now get initial guesses from the relative poses
    initial_guess_odom1_odom2 = transforms_dict["{}_{}".format(types2sensors[type_1],
                                                               types2sensors[type_2])]
    # Print out all the initial estimates as poses
    print("INITIAL GUESS {} {}: \n {} \n".format(types2sensors[type_1],
                                                 types2sensors[type_2],
                                                 initial_guess_odom1_odom2))

    # Get rotation matrices for initial guesses
    R0_odom1_odom2, t0_odom1_odom2 = initial_guess_odom1_odom2[:3, :3], \
                                     initial_guess_odom1_odom2[:3, 3]
    X0_odom1_odom2 = (R0_odom1_odom2, t0_odom1_odom2)  # Pymanopt estimate
    print("INITIAL GUESS {} {}: \n R0: \n {} \n\n t0: \n {} \n".format(
        types2sensors[type_1], types2sensors[type_2], R0_odom1_odom2, t0_odom1_odom2))

    # Create KFold xval object to get training/validation indices
    kf = KFold(n_splits=K, random_state=None, shuffle=False)
    k = 0  # Set fold counter to 0

    # Dataset
    A = np.array(odom2_rel_poses)  # First set of poses
    B = np.array(odom1_rel_poses)  # Second set of poses
    N = len(A)
    assert len(A) == len(B)  # Sanity check to ensure odometry data matches
    r = np.logical_or(np.array(odom1_reject)[:N], np.array(odom2_reject)[:N])  # Outlier rejection

    print("NUMBER OF CROSS-VALIDATION FOLDS: {}".format(K))

    # Iterate over 30 second intervals of the poses
    for train_index, test_index in kf.split(A):  # Perform K-fold cross-validation

        # Path for results from manifold optimization
        analysis_results_path = os.path.join(ANALYSIS_RESULTS_PATH, "k={}".format(k))
        final_estimates_path = os.path.join(FINAL_ESTIMATES_PATH, "k={}".format(k))
        odometry_plots_path = os.path.join(ODOMETRY_PLOTS_PATH, "k={}".format(k))

        # Make sure all paths exist - if they don't create them
        for path in [analysis_results_path, final_estimates_path, odometry_plots_path]:
            check_dir(path)

        # Get training data
        A_train = A[train_index]
        B_train = B[train_index]
        N_train = min(A_train.shape[0], B_train.shape[0])
        r_train = r[train_index]
        print("FOLD NUMBER: {}, NUMBER OF TRAINING SAMPLES: {}".format(k, N_train))

        omega = np.max([var_R_odom1, var_R_odom2])  # Take average across different odometries
        rho = np.max([var_t_odom1, var_t_odom2])  # Take average across different odometries

        cost_lambda = lambda x: cost(x, A_train, B_train, r_train, rho, omega, WEIGHTED)  # Create cost function
        problem = Problem(manifold=manifold, cost=cost_lambda)  # Create problem
        solver = CustomSteepestDescent()  # Create custom solver
        X_opt = solver.solve(problem, x=X0_odom1_odom2)  # Solve problem
        print("Initial Guess for Main-Front Transformation: \n {}".format(initial_guess_odom1_odom2))
        print("Optimal solution between {} and {} "
              "reference frames: \n {}".format(types2sensors[type_1],
                                               types2sensors[type_2],
                                               X_opt))

        # Take intermediate values for plotting
        estimates_x = solver.estimates
        errors = solver.errors
        iters = solver.iterations

        # Metrics dictionary
        estimates_dict = {i: T for i, T in zip(iters, estimates_x)}
        error_dict = {i: e for i, e in zip(iters, errors)}

        # Save intermediate results to a pkl file
        estimates_fname = os.path.join(analysis_results_path, "estimates_{}_{}.pkl".format(types2sensors[type_1],
                                               types2sensors[type_2], X_opt))
        error_fname = os.path.join(analysis_results_path, "error_{}_{}.pkl".format(types2sensors[type_1],
                                               types2sensors[type_2], X_opt))

        # Save estimates to pickle file
        with open(estimates_fname, "wb") as pkl_estimates:
            pickle.dump(estimates_dict, pkl_estimates)
            pkl_estimates.close()

        # Save error to pickle file
        with open(error_fname, "wb") as pkl_error:
            pickle.dump(error_dict, pkl_error)
            pkl_error.close()

        # Calculate difference between initial guess and final
        X_opt_T = construct_pose(X_opt[0], X_opt[1].reshape((3, 1)))
        print("DIFFERENCE IN MATRICES: \n {}".format(
            np.subtract(X_opt_T, initial_guess_odom1_odom2)))

        # Compute the weighted RMSE (training/in-sample)
        train_rmse_init_weighted, train_rmse_final_weighted, train_rmse_init_R_weighted, \
        train_rmse_init_t_weighted, train_rmse_final_R_weighted, \
        train_rmse_final_t_weighted = compute_rmse_weighted(
            initial_guess_odom1_odom2, X_opt_T, A_train, B_train, rho, omega)

        # Compute the unweighted RMSE (training/in-sample)
        train_rmse_init_unweighted, train_rmse_final_unweighted, train_rmse_init_R_unweighted, \
        train_rmse_init_t_unweighted, train_rmse_final_R_unweighted, \
        train_rmse_final_t_unweighted = compute_rmse_unweighted(
            initial_guess_odom1_odom2, X_opt_T, A_train, B_train)

        # Concatenate all RMSE values for training/in-sample
        train_rmses = [train_rmse_init_unweighted, train_rmse_final_unweighted,
                      train_rmse_init_weighted, train_rmse_final_weighted,
                      train_rmse_init_R_unweighted, train_rmse_init_t_unweighted,
                      train_rmse_final_R_unweighted, train_rmse_final_t_unweighted,
                      train_rmse_init_R_weighted, train_rmse_init_t_weighted,
                      train_rmse_final_R_weighted, train_rmse_final_t_weighted]

        # Display and save RMSEs
        outpath = os.path.join(analysis_results_path,
                               "train_rmse_{}_{}.txt".format(
                                   types2sensors[type_1],
                                   types2sensors[type_2]))
        display_and_save_rmse(train_rmses, outpath)

        # Get test data
        A_test = A[test_index]
        B_test = B[test_index]
        N_test = min(A_test.shape[0], B_test.shape[0])
        print("NUMBER OF TEST SAMPLES: {}".format(N_test))

        # Compute the weighted RMSE (testing/out-of-sample)
        test_rmse_init_weighted, test_rmse_final_weighted, test_rmse_init_R_weighted, \
        test_rmse_init_t_weighted, test_rmse_final_R_weighted, \
        test_rmse_final_t_weighted = compute_rmse_weighted(initial_guess_odom1_odom2,
                                                            X_opt_T, A_test, B_test, rho, omega)

        # Compute the unweighted RMSE (testing/out-of-sample)
        test_rmse_init_unweighted, test_rmse_final_unweighted, test_rmse_init_R_unweighted, \
        test_rmse_init_t_unweighted, test_rmse_final_R_unweighted, \
        test_rmse_final_t_unweighted = compute_rmse_unweighted(initial_guess_odom1_odom2,
                                                                X_opt_T, A_test, B_test)

        # Concatenate all RMSE values for testing/out-of-sample
        test_rmses = [test_rmse_init_unweighted, test_rmse_final_unweighted,
                 test_rmse_init_weighted, test_rmse_final_weighted,
                 test_rmse_init_R_unweighted, test_rmse_init_t_unweighted,
                 test_rmse_final_R_unweighted, test_rmse_final_t_unweighted,
                 test_rmse_init_R_weighted, test_rmse_init_t_weighted,
                 test_rmse_final_R_weighted, test_rmse_final_t_weighted]

        # Display and save RMSEs
        outpath = os.path.join(analysis_results_path,
                               "test_rmse_{}_{}.txt".format(types2sensors[type_1], types2sensors[type_2]))
        display_and_save_rmse(test_rmses, outpath)

        # Save final estimates
        final_estimate_outpath = os.path.join(final_estimates_path,
                                              "{}_{}.txt".format(types2sensors[type_1], types2sensors[type_2]))
        np.savetxt(final_estimate_outpath, X_opt_T)

        # Finally, increment k
        k += 1

def main():
    """Main function to run nonlinear manifold optimization on SE(3) to estimate
    an optimal relative pose transformation between coordinate frames given by
    the different lidar sensors."""
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


    # Calculate relative poses
    (main_aligned, main_rel_poses) = relative_pose_processing.calc_rel_poses(main_aligned)
    (front_aligned, front_rel_poses) = relative_pose_processing.calc_rel_poses(front_aligned)
    (rear_aligned, rear_rel_poses) = relative_pose_processing.calc_rel_poses(rear_aligned)

    cov_t_main, cov_R_main = compute_weights_euler(main_aligned)
    cov_t_front, cov_R_front = compute_weights_euler(front_aligned)
    cov_t_rear, cov_R_rear = compute_weights_euler(rear_aligned)

    # Extract a single scalar using the average value from rotation and translation
    var_t_main = extract_variance(cov_t_main, mode="max")
    var_R_main = extract_variance(cov_R_main, mode="max")
    var_t_front = extract_variance(cov_t_front, mode="max")
    var_R_front = extract_variance(cov_R_front, mode="max")
    var_t_rear = extract_variance(cov_t_main, mode="max")
    var_R_rear = extract_variance(cov_R_rear, mode="max")


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
    omega = np.max([var_R_main, var_R_front])  # Take average across different odometries
    rho = np.max([var_t_main, var_t_front])  # Take average across different odometries
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
    omega = np.max([var_R_main, var_R_rear])  # Take average across different odometries
    rho = np.max([var_t_main, var_t_rear])  # Take average across different odometries
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
    omega = np.max([var_R_front, var_R_rear])  # Take average across different odometries
    rho = np.max([var_t_front, var_t_rear])  # Take average across different odometries
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


def main_cross_val():
    """Main function to run when using cross-validation."""
    # Extract and process the CSVs
    main_odometry = relative_pose_processing.process_df(MAIN_ODOM_CSV)
    front_odometry = relative_pose_processing.process_df(FRONT_ODOM_CSV)
    rear_odometry = relative_pose_processing.process_df(REAR_ODOM_CSV)

    # Process poses
    (main_aligned, front_aligned, rear_aligned) = \
        relative_pose_processing.align_df([main_odometry,
                                           front_odometry,
                                           rear_odometry])

    # Now pick pairwise combinations of poses and run cross-validation
    odom_pairs = [(main_odometry, front_odometry),
                  (main_odometry, rear_odometry)]
    alignment_pairs = [(main_aligned, front_aligned),
                       (main_aligned, rear_aligned)]
    type_pairs = [("main", "front"),
                  ("main", "rear")]

    # Iterate over pairwise combinations of lidars
    for odom_pair, alignment_pair, type_pair in zip(odom_pairs, alignment_pairs, type_pairs):
        # Decompose data
        odom_1, odom_2 = odom_pair
        aligned_1, aligned_2 = alignment_pair
        type_1, type_2 = type_pair

        # Display
        print("RUNNING CROSS-VALIDATION FOR {} TO {} LIDAR".format(type_2,
                                                                   type_1))
        print("USING {} FOLDS".format(FOLDS))
        # Run cross-validation for each pair of lidars
        cross_validation(odom_1, aligned_1, odom_2, aligned_2,
                         type_1, type_2, K=FOLDS)

def parse_args():
    """Command-line argument parser.

    Returns:
        args: A parsed arguments object.
    """
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-thr", "--reject_threshold", type=int, default=100,
                        help="Value of maximum ICP diagonal with which to reject"
                             "a sample.")
    parser.add_argument("-kfolds", "--kfolds", type=int, default=10,
                        help="If cross-validation is enabled, this is the number "
                             "of cross-validation folds to use for it.")
    parser.add_argument("-use_xval", "--use_xval", action="store_true",
                         help="Whether to use cross-validation.")
    parser.add_argument("-weighted", "--weighted", action="store_true",
                        help="Whether to use weighted optimization to account"
                             "for noise in rotation and translation.")

    # Parse args and return them
    return parser.parse_args()

if __name__ == '__main__':

    # Parse args
    args = parse_args()
    print("ARGUMENT CONFIGURATION: {}".format(args))

    ##### FLAGS #####
    REJECT_THR = args.reject_threshold  # For rejecting samples if the ICP covariance is too high
    FOLDS = args.kfolds  # Cross-validation splits
    WEIGHTED = args.weighted  # Whether to weight our estimate
    USE_CROSS_VAL = args.use_xval  # Flag to denote whether we use cross-validation
    #################

    # Specify odometry CSV file paths

    MAIN_ODOM_CSV = os.path.join('data', 'main_odometry_clean.csv')
    FRONT_ODOM_CSV = os.path.join('data', 'front_odometry_clean.csv')
    REAR_ODOM_CSV = os.path.join('data', 'rear_odometry_clean.csv')

    # Specify path for relative transformations between poses
    PKL_POSES_PATH = os.path.join('data', 'relative_transforms.pkl')

    # Path for results from manifold optimization
    if USE_CROSS_VAL:
        ANALYSIS_RESULTS_PATH = os.path.join(
            'cross_validation_folds={}'.format(FOLDS),
            'analysis_results_weighted_{}'.format(WEIGHTED))
        FINAL_ESTIMATES_PATH = os.path.join(
            'cross_validation_folds={}'.format(FOLDS),
            'final_estimates_weighted_{}'.format(WEIGHTED))
        ODOMETRY_PLOTS_PATH = os.path.join(
            'cross_validation_folds={}'.format(FOLDS),
            'odometry_plots_weighted_{}'.format(WEIGHTED))
    else:
        ANALYSIS_RESULTS_PATH = os.path.join('results', 'analysis_results_weighted_{}'.format(WEIGHTED))
        FINAL_ESTIMATES_PATH = os.path.join('results', 'final_estimates_weighted_{}'.format(WEIGHTED))
        ODOMETRY_PLOTS_PATH = os.path.join('results', 'odometry_plots_weighted_{}'.format(WEIGHTED))

    # Check directories, and if they do not exist, make them
    for path in [ANALYSIS_RESULTS_PATH, FINAL_ESTIMATES_PATH,
                 ODOMETRY_PLOTS_PATH]:
        check_dir(path)

    if USE_CROSS_VAL:  # Run cross-validation
        main_cross_val()
    else:
        main()
