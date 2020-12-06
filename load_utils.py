"""Script for loading data from our various pickle and CSV files."""

import os
import pickle

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




my_dict = load_pkl(os.path.join("analysis_results", "estimates_front_rear.pkl"))
print("MY DICT: \n {}".format(my_dict))