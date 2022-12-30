import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes and returns the accurancy score of the model on a given dataset
    :param y_true: np.ndarray - The true values of the labels
    :param y_pred: np.ndarray - the labels predicted by a classifier
    :return:
    """
    # calculates the number of correct predictions by comparing the true labels with the predicted labels
    # negative, true positive, false negative and false positive

    return np.sum((y_true == y_pred) / len(y_true))
