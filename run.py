# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from utils.helpers import *
from utils.prediction import *
from utils.preprocess import *
from utils.cross_validation import *


class Args(object):
    """Define hyperparameters."""

    def __init__(self):
        """Model arguments.

        Args:
            train_path: path for training data.
            test_path: path for test data.
            submission_path: path for submission.
            degree: polynomial term.
            lambda_: penalizing term.
            k_fold: we use 1/k_fold of the training data as the validation data.
            seed: random seed indicating the deadline of the project.
        """
        self.train_path = "./data/train.csv"
        self.test_path = "./data/test.csv"
        self.submission_path = "./data/submission_final.csv"
        self.degree = 9
        self.lambda_ = 1e-8
        self.k_fold = 5
        self.seed = 20221031


if __name__ == "__main__":

    args = Args()

    # Load data.
    y_raw_tr, tx_raw_tr, ids_tr = load_csv_data(args.train_path)
    _, tx_raw_te, ids_te = load_csv_data(args.test_path)

    # Cross validation.
    k_indices = build_k_indices(y_raw_tr, args.k_fold, args.seed)
    tx_raw_tr, tx_raw_dev, y_raw_tr, y_raw_dev = cross_validation_dataset(
        y_raw_tr, tx_raw_tr, k_indices, k=args.k_fold - 1
    )

    # Feature engineering
    (
        tx_tr_list,
        y_tr_list,
        tx_dev_list,
        y_dev_list,
        tx_te,
        maxs,
        mins,
        means,
        stds
    ) = feature_engineering(
        y_raw_tr, y_raw_dev, tx_raw_tr, tx_raw_dev, tx_raw_te, args.degree
    )

    # Training
    ws = []
    y_tr_pred, y_tr_true = np.empty((0, 1)), np.empty((0, 1))
    y_dev_pred, y_dev_true = np.empty((0, 1)), np.empty((0, 1))

    for i in range(len(tx_tr_list)):

        y_tr = y_tr_list[i]
        tx_tr = tx_tr_list[i]
        y_dev = y_dev_list[i]
        tx_dev = tx_dev_list[i]

        best_w, train_loss, dev_loss = ridge_regression_cv(
            y_tr,
            tx_tr,
            y_dev,
            tx_dev,
            args.lambda_
        )

        y_tr_pred = np.vstack((y_tr_pred, predict_linear(tx_tr, best_w)))
        y_dev_pred = np.vstack((y_dev_pred, predict_linear(tx_dev, best_w)))
        y_tr_true = np.vstack((y_tr_true, y_tr))
        y_dev_true = np.vstack((y_dev_true, y_dev))
        ws.append(best_w)

    # Report metrics.
    accuracy, precision, recall, f1_score = compute_metrics(y_tr_true, y_tr_pred)
    print("==========Training==========")
    print("Acc={}; P={}; R={}; F1={}".format(accuracy, precision, recall, f1_score))

    accuracy, precision, recall, f1_score = compute_metrics(y_dev_true, y_dev_pred)
    print("==========Validation========")
    print("Acc={}; P={}; R={}; F1={}".format(accuracy, precision, recall, f1_score))

    # Submit testing results.
    D = tx_te.shape[1]
    y_pred = np.empty((0, 1))
    for i in range(len(tx_te)):
        pri_jet_num = np.min((2, int(tx_te[i, -1])))
        w = ws[pri_jet_num]

        if pri_jet_num == 0:
            col_mask = np.ones(D, dtype=bool)
            col_mask[[4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]] = False
            tx_cleaned = tx_te[i, col_mask].reshape(1, -1)
        elif pri_jet_num == 1:
            col_mask = np.ones(D, dtype=bool)
            col_mask[[4, 5, 6, 12, 26, 27, 28, 29]] = False
            tx_cleaned = tx_te[i, col_mask].reshape(1, -1)
        else:
            tx_cleaned = tx_te[i, :-1].reshape(1, -1)

        mean = means[pri_jet_num]
        std = stds[pri_jet_num]
        tx_cleaned = np.clip(tx_cleaned, mean-2*std, mean+2*std)
        tx_cleaned = build_poly(tx_cleaned, args.degree)
        tx = (
            (tx_cleaned - mins[pri_jet_num]) / (maxs[pri_jet_num] - mins[pri_jet_num])
        ).reshape(1, -1)
        y_pred = np.vstack((y_pred, predict_linear(tx, w)))

    y_pred[y_pred == 0] = -1
    y_pred = y_pred.astype(int)

    create_csv_submission(ids_te, y_pred, args.submission_path)
