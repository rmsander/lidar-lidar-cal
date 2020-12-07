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
    other for translation estimates."""

    # Load data from pkl file
    pkl_dict = load_pkl(pkl_file)
    max_iter = max(list(pkl_dict.keys()))

    # Take rotation and translation estimates separately
    X_rot = np.array([pkl_dict[i][0] for i in range(1, max_iter + 1)])
    X_trans = np.array([pkl_dict[i][1] for i in range(1, max_iter + 1)])


    # Save translation and rotation
    out_fname = pkl_file.split(".")[0] + "_{}" + ".txt"
    #np.savetxt(out_fname.format("R"), X_rot)
    #np.savetxt(out_fname.format("t"), X_trans)

    with open(out_fname.format("R"), 'w') as R_txt_file:
        for i in range(0, max_iter):
            R_txt_file.write(str(X_rot[i]))
            R_txt_file.write("\n")

    with open(out_fname.format("t"), 'w') as t_txt_file:
        for i in range(0, max_iter):
            t_txt_file.write(str(X_trans[i]))
            t_txt_file.write("\n")

def convert_to_txt():
    """Function to convert pickle files with incremental estimates to txt files."""
    try:
        # Take paths from pkl
        path_main_front = os.path.join("analysis_results", "estimates_main_front.pkl")
        path_main_rear = os.path.join("analysis_results", "estimates_main_rear.pkl"
        path_front_rear = os.path.join("analysis_results", "estimates_front_rear.pkl")

        # Save them to text path
        save_to_txt(path_main_front)
        save_to_txt(path_main_rear)
        save_to_txt(path_front_rear)

    except:
        print("PATH NOT FOUND")




