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
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    """
       Computes the derivative of the Mean Squared Error (MSE) function.
       :param y_true: The true labels of the dataset.
       :param y_pred: The predicted labels of the dataset.
       :return: The derivative of the MSE function.
    """
    return -2 * (y_true -y_pred) / (len(y_true) * 2)


