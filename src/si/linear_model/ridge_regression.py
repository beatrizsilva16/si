import numpy as np
import sys
PATHS = ["src/si/data/dataset", "../metrics"]
sys.path.extend(PATHS)
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression:

    """
    Implements Ridge Regression, a linear model using the L2 regularization. This model solves the
    linear regression problem using an adapted Gradient Descent technique.
    """

    def __init__(self,
                 l2_penalty: float = 1,
                 alpha: float = 0.001,
                 max_iter: int = 1000):
        """
        Implements Ridge Regression, a linear model using the L2 regularization. This model solves the
        linear regression problem using an adapted Gradient Descent technique.
        Parameters
        ----------
        l2_penalty: int, float (default=1)
            The L2 regularization coefficient
        alpha: int, float (default=0.001)
            The learning rate
        max_iter: int (default=1000)
            The maximum number of iterations
        tolerance: int, float (default=1)
            Tolerance for stopping gradient descent (maximum absolute difference in the value of the
            loss function between two iterations)
        adaptative_alpha: bool (default=False)
            Whether an adaptative alpha is used in the gradient descent
        Attributes
        ----------
        fitted: bool
            Whether the model is already fitted
        theta: np.ndarray
            Model parameters, namely the coefficients of the linear model
        theta_zero: float
            Model parameter, namely the intercept of the linear model
        cost_history: dict
            A dictionary containing the values of the cost function (J function) at each iteration
            of the algorithm (gradient descent)
        """

        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = None

    def fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        It fits the model to the dataset.
        :param dataset: Dataset, dataset to fit the model to
        :return: self, fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)  # new array de dimension n filled with zeros
        self.theta_zero = 0

        # cost history
        self.cost_history = {}  # dic empty

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero  # function y = mx + b

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)
            # calculates the gradient of the cost function
            # np.dot sums the colum values of the multiplication arrays
            # learning rate is multiplicated by 1/m to normalize the rate to the dataset size

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # computes the cost function
            self.cost_history[i] = self.cost(dataset)

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # Exercise 6.3 - When the difference between the cost of the previous iteration and the cost of the
            # current iteration is less than 1, the Gradient Descend should stop.
            if i > 0:
                if self.cost_history[i - 1] - self.cost_history[i] < 1:
                    break

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        It predicts the output of the dataset.
        :param dataset: Dataset, dataset to predict the output of
        :return: np.ndarray, the predictions of the dataset
        """
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        It computes the Mean Square Error (MSE) of the model on the dataset
        :param dataset: Dataset, dataset to compute the MSE on
        :return: float, the MSE value of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost function (J function) of the model on the dataset using L2 regularization.
        :param dataset: Dataset, dataset to compute the cost function on
        :return: float, the cost function of the model
        """
        y_pred = self.predict(dataset)
        cost_func = (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (
                    2 * len(dataset.y))
        return cost_func

    # Exercise 6.2. - add a method that computes a plot of the cost function history of the model
    def cost_function_plot(self):
        """
        It plots the cost function history of the model.
        :return: None
        """
        import matplotlib.pyplot as plt

        # plot - Y axis should contain the cost value while the X axis should contain the iterations
        x_iterations = list(self.cost_history.keys())
        y_values = list(self.cost_history.values())

        # plot construction
        plt.plot(x_iterations, y_values, '-r')
        plt.title("Cost History of the model")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()


