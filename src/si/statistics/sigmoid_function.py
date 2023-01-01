import numpy as np

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    Computes and returns the sigmoid function of the given input.
    Parameters
    ----------
    X: np.ndarray
        The input of the sigmoid function
    """
    return 1 / (1 + np.exp(-X))