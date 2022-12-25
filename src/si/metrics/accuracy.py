import numpy as np

def accurancy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes and returns the accurancy score of the model on a given dataset
    :param y_true: np.ndarray - The true values of the labels
    :param y_pred: np.ndarray - the labels predicted by a classifier
    :return:
    """

    return np.sum (y_true == y_pred) / len(y_true)

