# -*- coding: utf-8 -*-
# data preprocessing
import numpy as np


def process_y(y_raw):
    """Change y-label -1 to 0.
    Args:
        y_raw: numpy array of shape (N, ) with label 1 and -1.
        
    Returns:
        y: numpy array of shape (N, 1) with label 1 and 0.
    """
    y_raw[y_raw==-1] = 0
    y_raw = y_raw.reshape(-1, 1)
    y = y_raw
    
    return y


def process_tx(tx_raw):
    """Remove some features with missing values and swap pri_jet_num to the last column.
    Args:
        tx_raw: numpy array of shape (N, 30)
        
    Returns:
        tx_cleaned: numpy array of shape (N, D). D is the reduced dimension.
    """
    col_mask = np.ones(tx_raw.shape[1], dtype=bool)
    col_mask[0] = False
    tx_cleaned = tx_raw[:, col_mask]
    tx_cleaned[:, [21, -1]] = tx_cleaned[:, [-1, 21]]
    
    return tx_cleaned


def standardization(tx_tr, tx_te):
    """Standardize the input by (X-mu) / sigma.
       We compute mean and std from train and test sets.
    Args:
        tx_tr: numpy array of shape (N_tr, D), D is the number of features.
        tx_te: numpy array of shape (N_te, D), D is the number of features.

    Returns:    
        tx_stdtr: standardized numpy array of shape (N_tr, D), D is the number of features.
        tx_stdte: standardized numpy array of shape (N_te, D), D is the number of features.
    """

    tx_all = np.vstack((tx_tr, tx_te))
    tx_mean = np.mean(tx_all, axis=0)
    tx_std = np.std(tx_all, axis=0)
    eps = 1e-9  # avoid being divided by zero

    tx_stdtr = (tx_tr - tx_mean) / (eps + tx_std)
    tx_stdte = (tx_te - tx_mean) / (eps + tx_std)

    return tx_stdtr, tx_stdte, tx_mean, tx_std


def normalization(tx_tr, tx_te):
    tx_all = np.vstack((tx_tr, tx_te))
    tx_max = np.max(tx_all, axis=0)
    tx_min = np.min(tx_all, axis=0)

    tx_stdtr = (tx_tr - tx_min) / (tx_max - tx_min)
    tx_stdte = (tx_te - tx_min) / (tx_max - tx_min)

    return tx_stdtr, tx_stdte, tx_max, tx_min


def log_plus_one(tx_tr, tx_te):
    tx_all = np.vstack((tx_tr, tx_te))
    tx_min = np.min(tx_all, axis=0)

    tx_logtr = np.log(tx_tr - tx_min + 1)
    tx_logte = np.log(tx_te - tx_min + 1)

    return tx_logtr, tx_logte


def split_jet_num(tx, y):
    """Split tx into three subsets with different pri_jet_num and remove columns with missing values for each jet num.

    Args:
        tx: numpy array of shape=(N, D)
        y: numpy array of shape=(N, 1)

    Returns:
        tx_list, y_list: two list of numpy arrays related to different pri_jet_num
    """
    D = tx.shape[1]

    tx_list = [tx[tx[:, -1] == i, :-1] for i in range(2)]
    y_list = [y[tx[:, -1] == i] for i in range(2)]
    tx_list.append(tx[tx[:, -1] >= 2, :-1])
    y_list.append(y[tx[:, -1] >= 2])

    # 1. pri_jet_num = 0
    col_mask = np.ones(D-1, dtype=bool)
    col_mask[[3, 4, 5, 11, 21, 22, 23, 24, 25, 26, 27]] = False
    tx_list[0] = tx_list[0][:, col_mask]

    # 2. pri_jet_num = 1
    col_mask = np.ones(D-1, dtype=bool)
    col_mask[[3, 4, 5, 11, 25, 26, 27]] = False
    tx_list[1] = tx_list[1][:, col_mask]

    return tx_list, y_list


def build_poly(tx, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        tx: numpy array of shape (N, D), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N, degree*D)
    """
    
    N, D = tx.shape
    poly = np.empty((N, 0))
    for i in range(degree):
        poly = np.hstack((poly, tx**(i+1)))
    
    return poly