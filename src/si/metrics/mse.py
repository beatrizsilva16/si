import numpy as np


def mse(y_true: np.ndarray, y_pred: np.array) -> float:
    """
    Returns the man squared error of the model on the given dataset
    :param y_true: np.ndarray
        The true labels of the dataset
    :param y_pred: np.ndarray
        The predicted labels of the dataset
    :return: mse: float
        The mean squared error of the model
    """
    N = y_true.shape[0]
    return np.sum((y_true - y_pred) ** 2) / (2 * N)


def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
       Computes the derivative of the Mean Squared Error (MSE) function.
       :param y_true: The true labels of the dataset.
       :param y_pred: The predicted labels of the dataset.
       :return: The derivative of the MSE function.
    """
    N = y_true.shape[0]
    return -(y_true - y_pred) / N



