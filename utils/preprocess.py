# -*- coding: utf-8 -*-
# data preprocessing
import numpy as np


def feature_engineering(y_raw_tr, y_raw_dev, tx_raw_tr, tx_raw_dev, tx_raw_te, degree):
    """Feature engineering.
    Args:
        y_raw_tr: numpy array of raw training label
        y_raw_dev: numpy array of raw validation label
        tx_raw_tr: numpy array of raw training data
        tx_raw_dev: numpy array of raw validation data
        tx_raw_te: numpy array of raw testing data
        degree: scalar, polynomial degree

    Returns:
        tx_tr_list: list of training data with different PRI_JET_NUM
        y_tr_list: list of label corresponding to tx_tr_list
        tx_dev_list: list of validation data with different PRI_JET_NUM
        y_dev_list: list of label corresponding to tx_dev_list
        tx_te: processed test data
        maxs: list of maxs of the processed training and validation data
        mins: list of mins of the processed training and validation data
        means: list of means of the processed training and validation data
        stds: list of stds of the processed training and validation data
    """
    # change -1 to 0
    y_tr = process_y(y_raw_tr)
    y_dev = process_y(y_raw_dev)

    # swap pri_jet_num to the last row
    # and fill -999 in the first column with nanmedian
    tx_tr = process_tx(tx_raw_tr)
    tx_dev = process_tx(tx_raw_dev)
    tx_te = process_tx(tx_raw_te)
    median = np.nanmedian(np.hstack((tx_tr[:, 0], tx_te[:, 0])))
    tx_tr[np.isnan(tx_tr[:, 0]), 0] = median
    tx_dev[np.isnan(tx_dev[:, 0]), 0] = median
    tx_te[np.isnan(tx_te[:, 0]), 0] = median

    # split datasets to different jet nums
    # and remove columns with missing values for each jet num
    tx_tr_list, y_tr_list = split_jet_num(tx_tr, y_tr)
    tx_dev_list, y_dev_list = split_jet_num(tx_dev, y_dev)
    split_number = len(tx_tr_list)

    # remove outliers
    means = []
    stds = []
    for i in range(3):
        mean = np.mean(np.vstack((tx_tr_list[i], tx_dev_list[i])), axis=0)
        std = np.std(np.vstack((tx_tr_list[i], tx_dev_list[i])), axis=0)
        tx_tr_list[i] = np.clip(tx_tr_list[i], mean - 2 * std, mean + 2 * std)
        tx_dev_list[i] = np.clip(tx_dev_list[i], mean - 2 * std, mean + 2 * std)
        means.append(mean)
        stds.append(std)

    # resampling
    for i in range(3):
        tx_tr_list[i], y_tr_list[i] = resample(tx_tr_list[i], y_tr_list[i], ifresample = True, redistribute = "over")

    # add polynomial features
    for i in range(split_number):
        tx_tr_list[i] = build_poly(tx_tr_list[i], degree)
        tx_dev_list[i] = build_poly(tx_dev_list[i], degree)

    # standardization
    maxs = [0 for i in range(split_number)]
    mins = [0 for i in range(split_number)]
    for i in range(split_number):
        tx_tr_list[i], tx_dev_list[i], maxs[i], mins[i] = normalization(
            tx_tr_list[i], tx_dev_list[i]
        )

    return (
        tx_tr_list,
        y_tr_list,
        tx_dev_list,
        y_dev_list,
        tx_te,
        maxs,
        mins,
        means,
        stds,
    )


def process_y(y_raw):
    """Change y-label -1 to 0.
    Args:
        y_raw: numpy array of shape (N, ) with label 1 and -1.

    Returns:
        y: numpy array of shape (N, 1) with label 1 and 0.
    """
    y_raw[y_raw == -1] = 0
    y_raw = y_raw.reshape(-1, 1)
    y = y_raw

    return y


def process_tx(tx_raw):
    """Fill the missing values for the first column and swap pri_jet_num to the last column.
    Args:
        tx_raw: numpy array of shape (N, 30)

    Returns:
        tx_cleaned: numpy array of shape (N, D). D is the reduced dimension.
    """
    tx_raw[:, [22, 29]] = tx_raw[:, [29, 22]]
    tx_raw[tx_raw[:, 0] == -999, 0] = np.nan
    tx_cleaned = tx_raw

    return tx_cleaned


def process_tx2(tx_raw):
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


def split_jet_num2(tx, y):
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
    col_mask = np.ones(D - 1, dtype=bool)
    col_mask[[3, 4, 5, 11, 21, 22, 23, 24, 25, 26, 27]] = False
    tx_list[0] = tx_list[0][:, col_mask]

    # 2. pri_jet_num = 1
    col_mask = np.ones(D - 1, dtype=bool)
    col_mask[[3, 4, 5, 11, 25, 26, 27]] = False
    tx_list[1] = tx_list[1][:, col_mask]

    return tx_list, y_list


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
    col_mask = np.ones(D - 1, dtype=bool)
    col_mask[[4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28]] = False
    tx_list[0] = tx_list[0][:, col_mask]

    # 2. pri_jet_num = 1
    col_mask = np.ones(D - 1, dtype=bool)
    col_mask[[4, 5, 6, 12, 26, 27, 28]] = False
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
        poly = np.hstack((poly, tx ** (i + 1)))

    return poly


def resample(X, Y, ifresample = False, redistribute = "False", noise = "False", *args):
    """ Resampling data by Y labels.
    
        Args:
        X: data,            np.array(shape = (N,D))
        Y: 2-value labels,  np.array(shape = (N)) 
        resample: bool
        redistribute: (optional)
            "False": resampled data have the same distribution with original data
            "over":  oversample X with minor label to match with major label
            "under": undersample X with majoy label to match size with minor label
        noise:        (optional) 
            "False": no adding noise to new X data
            "Gaussian: add a tuned noise e~N(0,1)^, amplified by eta given in *args
        args: optional parameters for noise
            if Gaussian:  args[0] == noise_level, array of float, size D or 1

        Return:
        X_new, Y_new: np arrays of resampled data
    """

    N, D = X.shape
    redis_param = ['False', 'over', 'under']
    noise_param = ['False', 'Gaussian']
    more = 1
    assert redistribute in redis_param and noise in noise_param, \
        "Illegal resampling argument error"

    if ifresample==True:
        if redistribute == "False":
            idx = np.random.randint(0, N, size = more*N)
            X_new = X[idx]
            Y_new = Y[idx]
        else:
            y_values = np.unique(Y)
            a_idxes = np.argwhere(Y==y_values[0]).flatten()
            b_idxes = np.argwhere(Y==y_values[1]).flatten()
            if (a_idxes.size < b_idxes.size):
                a_idxes, b_idxes = b_idxes, a_idxes
            if redistribute == "over":
                sample_size = a_idxes.size
            elif redistribute == "under":
                sample_size = b_idxes.size
        
            idx_a = a_idxes[np.random.randint(0, a_idxes.size, sample_size*more),]
            idx_b = b_idxes[np.random.randint(0, b_idxes.size, sample_size*more)]

            X_new = np.concatenate((X[idx_a, :], X[idx_b,:]), axis = 0)
            Y_new = np.concatenate((Y[idx_a], Y[idx_b]), axis = 0)
    else: 
        X_new, Y_new = X, Y

    if noise == "False":
        return X_new, Y_new
    elif noise == "Gaussian":
        noise_level = args[0]  
        # print(len(noise_level))
        assert (len(noise_level)==1 or len(noise_level)== D),\
            "noise level dimention not match with X"
        noise = np.random.normal(size=X_new.shape) * noise_level
        X_new = np.einsum("ni, ni -> ni", X_new, noise)
        return X_new, Y_new
