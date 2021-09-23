import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats as scs
from itertools import combinations

# needed for the loading bar
def is_notebook():
    """Distinguishes notebook from sheel for loading bar"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
# ----------------------------------------------------------------------------------------------------------------------

def chi2r(X, Y, dX, dY):
    """
    Compute reduced chi2 from data
    """
    if len(X.shape) == 2:
        N = X.shape[0]*X.shape[1]
    else:
        N = X.shape[0]
    err2 = dX**2 + dY**2
    return np.sum((X - Y)**2 / err2) / N
# ----------------------------------------------------------------------------------------------------------------------

def chi2(X, Y, dX, dY):
    """Compute chi2"""
    err2 = dX ** 2 + dY ** 2
    return np.sum((X - Y) ** 2 / err2)
# ----------------------------------------------------------------------------------------------------------------------

def rmsd(X, Y):
    """
    Compute RMSD from data
    """
    if len(X.shape) == 2:
        N = X.shape[0]*X.shape[1]
    else:
        N = X.shape[0]
    return np.sqrt(np.sum((X - Y)**2) / N)
# ----------------------------------------------------------------------------------------------------------------------

def create_masks(mx, intervals):
    """
    Build a binary mask from data
    """
    mask = np.empty(mx)
    mask.fill(1)
    for i in intervals:
        mask[i[0]:i[1]].fill(0)
    return mask, 1 - mask
# ----------------------------------------------------------------------------------------------------------------------

def save_pickle(to_save, filename):
    """
    Save an object as a pickle file

    Input
    -----
    to_save : any
        object to be saved as a pickle
    """
    with open(filename, "wb") as fp:
        pickle.dump(to_save, fp)
# ------------------------------------------------------------------------------------------------------------------

def load_pickle(to_load):
    """
    Load a pickle file with results.

    Parameters
    ----------
    to_load : str
        path to the file.
    """
    pin = open(to_load, "rb")
    loaded = pickle.load(pin)
    return loaded
# ------------------------------------------------------------------------------------------------------------------

def arg_quantile(h, q):
    """Returns the index of an array with a distribution corresponding to a given quantile"""
    sh = np.sort(h)
    idx = q * (len(sh) - 1)
    idx = int(idx + 0.5)
    idx = np.argpartition(sh, idx)[idx]

    return idx
# ------------------------------------------------------------------------------------------------------------------
