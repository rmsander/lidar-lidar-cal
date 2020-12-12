#!/usr/bin/python3

"""Functions for loading data from our various pickle and CSV files."""

# Native Python imports
import os
import pickle

# NumPy
import numpy as np


def load_pkl(f):
    """Loads a pkl file containing a serialized dictionary from memory.

    Parameters:
        f (str): A string denoting the path to the file to be loaded.  Path
            can be either absolute or relative.path

    Returns:
        pkl_dict (dict): A dictionary loaded from the serialized pickle file.
    """
    with open(f, "rb") as pkl_file:
        pkl_dict = pickle.load(pkl_file)
        pkl_file.close()

    return pkl_dict


def save_to_txt(pkl_file):
    """Converts a pkl file two text files - one for rotation estimates, and the
    other for translation estimates.

    Parameters:
        pkl_file (str):  String corresponding to the filepath of the pkl file
            containing the estimates of rotation and translation for each
            optimization step.
    """
    # Load data from pkl file
    pkl_dict = load_pkl(pkl_file)
    max_iter = max(list(pkl_dict.keys()))

    # Take rotation and translation estimates separately
    X_rot = np.array([pkl_dict[i][0] for i in range(1, max_iter + 1)])
    X_trans = np.array([pkl_dict[i][1] for i in range(1, max_iter + 1)])

    # Save translation and rotation
    out_fname = pkl_file.split(".")[0] + "_{}" + ".txt"

    # Save rotation estimates from each optimization step to a text file
    with open(out_fname.format("R"), 'w') as R_txt_file:
        for i in range(0, max_iter):
            R_txt_file.write(str(X_rot[i]))
            R_txt_file.write("\n")

    # Save translation estimates from each optimization step to a text file
    with open(out_fname.format("t"), 'w') as t_txt_file:
        for i in range(0, max_iter):
            t_txt_file.write(str(X_trans[i]))
            t_txt_file.write("\n")


def convert_to_txt(estimates_dir):
    """Function to convert pickle files with incremental estimates to txt files.

    Parameters:
        estimates_dir (str):  String corresponding to the name of the directory
            where estimates are stored.
    """
    try:
        # Take paths from pkl
        path_main_front = os.path.join(estimates_dir, "estimates_main_front.pkl")
        path_main_rear = os.path.join(estimates_dir, "estimates_main_rear.pkl")
        path_front_rear = os.path.join(estimates_dir, "estimates_front_rear.pkl")

        # Save them to text path
        save_to_txt(path_main_front)
        save_to_txt(path_main_rear)
        save_to_txt(path_front_rear)

    except:
        print("PATH NOT FOUND")


def check_dir(dir_path):
    """Function to check whether a given input path is a valid directory.

    Parameters:
        dir_path (str):  String denoting the path to check directory status. If
            the provided path is not a directory, then the directory is created
            as a sub-directory of the current path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print("Directory created: {}".format(dir_path))


def average_kfold_rmse(base_path, K=10):
    """Function to average the in-sample and out-of-sample RMSE for the optimal
    pose estimates.

    Parameters:
        base_path (str):  String corresponding to the base path where results are 
            stored.
        K (int):  The number of folds considered.
    """
    rmse_results = {}  # Initialize results dict
    for k in range(K):  # Iterate over folds
        path = base_path.format(k)
        with open(path, "r") as res_file:
            for line in res_file.readlines():
                pair = line.split(":")
                rmse_type, val = pair[0], pair[1]
                if k == 0:  # If first fold, get keys and values, else, just get values
                    rmse_results[rmse_type] = [float(val)]
                else:
                    rmse_results[rmse_type].append(float(val))
            res_file.close()
            
    # Now average the results
    rmse_avg_results = {}
    for key, val in rmse_results.items():
        rmse_avg_results[key] = np.mean(val)
    
    # Now save averaged results
    out_path = os.path.join("results", "cross_validation_folds={}".format(K),
                            "averaged_rmse_results.pkl")
    with open(out_path, "wb") as pkl_res:
        pickle.dump(rmse_avg_results, pkl_res)
        pkl_res.close()
    
    # Now print out the results
    for key, val in rmse_avg_results.items():
        print("{}: {:.4e}".format(key, val))

path = os.path.join("results", "cross_validation_folds=5",
                    "analysis_results_weighted_True", "k={}",
                    "test_rmse_velodyne_front.txt")
average_kfold_rmse(path, K=5)