import numpy as np

# Exercise 11 - 11.1. Add new error metric cross-entropy


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the cross-entropy error of the model on the given dataset.
    The cross-entropy is calculated by the difference of two probabilities, the true values and the predicted ones.
    :param y_true: np.ndarray, true labels of the dataset
    :param y_pred: np.ndarray, predicted labels of the dataset
    :return: float, the cross-entropy
    """
    # cross-entropy formula: - np.sum((y_true * np.log(y_pred))/ n
    # n = number of samples
    n = len(y_true) # could also use n = y_true.shape[0]
    # formula cross-entropy
    formula_ce = - np.sum(y_true * np.log(y_pred)) / n
    return formula_ce

# Exercise 11.2. Add the cross-entropy derivative


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    """
    It returns the derivative of the cross entropy function of the model on the given dataset.
    :param y_true: np.ndarray, true labels of the dataset
    :param y_pred: np.ndarray, predicted labels of the dataset
    :return: float, the cross-entropy derivative
    """
    n = len(y_true)
    formula_d_ce = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / n
    return formula_d_ce


if __name__ == "__main__":
    true_l = np.array([0,1,1,1,0,1])
    predict_l = np.array([0.9,0.1,0.1,0.9,0.1,0.9])
    # cross-Entropy
    print(f"Cross-Entropy: {cross_entropy(true_l, predict_l):.4f}")
    # cross-Entropy derivative
    print(f"d(Cross-Entropy)/d(y_pred): {cross_entropy_derivative(true_l, predict_l)}")
