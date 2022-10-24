# -*- coding: utf-8 -*-
# functions for prediction
import numpy as np
from implementations import *


def predict_linear(tx, w, threshold=0.5):
    """Predict values from linear regression. If probability > threshold, predict 1.
    Args:
        tx: numpy array of shape (N, D), D is the number of features.
        w: optimal weights, numpy array of shape(D, 1), D is the number of features.
        threshold: scalar, default = 0.5, threshold for decision.

    Returns:
        y_pred: numpy array of shape (N, 1) with labels 1 or 0, N is the number of samples.
    """

    y_pred = tx @ w
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    return y_pred


def predict_logistic(tx, w, threshold=0.5):
    """Predict values from logistic regression. If probability > threshold, predict 1.
    Args:
        tx: numpy array of shape (N, D), D is the number of features.
        w: optimal weights, numpy array of shape(D, 1), D is the number of features.
        threshold: scalar, default = 0.5, threshold for decision.

    Returns:
        y_pred: numpy array of shape (N, 1) with labels 1 or 0, N is the number of samples.
    """

    y_pred = sigmoid(tx @ w)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    return y_pred


def compute_metrics(y, y_pred):
    """Compute prediction accuracy, f1-score, precision, recall.
    Args:
        y: numpy array of shape (N, 1), the true label.
        y_pred: numpy array of shape (N, 0), the prediction.

    Returns:
        accuracy: the scalar of accuracy.
        f1_score: the scalar of f1_score.
        precision: the scalar of precision.
        recall: the scalar of recall.
    """

    accuracy = (y == y_pred).sum() / y.shape[0]
    precision = ((y == y_pred) & (y == 1)).sum() / y_pred.sum()
    recall = ((y == y_pred) & (y == 1)).sum() / y.sum()
    f1_score = 2 / (1 / precision + 1 / recall)
    return accuracy, precision, recall, f1_score
