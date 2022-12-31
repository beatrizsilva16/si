from cmath import sqrt

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculated the Root Mean Squared error.
    :param y_true: An array of true labels.
    :param y_pred: An array of predicted labels.
    :return: The RMSE value.
    """

    N = y_true.shape[0]  # N represents the number of samples
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / N)  # RMSE formula