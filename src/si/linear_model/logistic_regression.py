import numpy as np
import sys
PATHS = ["src/si/data", "src/si/metrics", "src/si/statistics"]
sys.path.extend(PATHS)
from dataset import Dataset
from accuracy import accuracy
from sigmoid_function import sigmoid_function
from typing import Union


class LogisticRegression:

    """
    Implements LogisticRegression, a logistic model using the L2 regularization. This model solves the
    logistic regression problem using an adapted Gradient Descent technique.
    """

    def __init__(self,
                 l2_penalty: Union[int, float] = 1,
                 alpha: Union[int, float] = 0.001,
                 max_iter: int = 1000,
                 tolerance: Union[int, float] = 0.0001,
                 adaptative_alpha: bool = False):
        """
        Implements LogisticRegression, a logistic model using the L2 regularization. This model solves the
        logistic regression problem using an adapted Gradient Descent technique.
        Parameters
        ----------
        l2_penalty: int, float (default=1)
            The L2 regularization coefficient
        alpha: int, float (default=0.001)
            The learning rate
        max_iter: int (default=1000)
            The maximum number of iterations
        tolerance: int, float (default=0.0001)
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
        # check values of parameters
        self._check_init(l2_penalty, alpha, max_iter, tolerance)
        # parameters
        self.l2_penalty = l2_penalty # lambda
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.adaptative_alpha = adaptative_alpha
        # attributes
        self.fitted = False
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    @staticmethod
    def _check_init(l2_penalty: Union[int, float],
                    alpha: Union[int, float],
                    max_iter: int,
                    tolerance: Union[int, float]):
        """
        Checks the values of numeric parameters.
        Parameters
        ----------
        l2_penalty: int, float
            The L2 regularization coefficient
        alpha: int, float
            The learning rate
        max_iter: int
            The maximum number of iterations
        tolerance: int, float
            Tolerance for stopping gradient descent (maximum absolute difference in the value of the
            loss function between two iterations)
        """
        if l2_penalty <= 0:
            raise ValueError("The value of 'l2_penalty' must be positive.")
        if alpha <= 0:
            raise ValueError("The value of 'alpha' must be positive.")
        if max_iter < 1:
            raise ValueError("The value of 'max_iter' must be a positive integer.")
        if tolerance <= 0:
            raise ValueError("The value of 'tolerance' must be positive.")

    def _gradient_descent_iter(self, dataset: Dataset, m: int) -> None:
        """
        Performs one iteration of the gradient descent algorithm. The algorithm goes as follows:
        1. Predicts the outputs of the dataset
            -> sigmoid(X @ theta + theta_zero)
        2. Computes the gradient vector and adjusts it according to the value of alpha
            -> -(alpha / m) * [(y_true / y_pred) @ X - ((1 - y_true) / (1 - y_pred)) @ X]
        3. Computes the penalization term for theta
            -> theta * alpha * (l2 / m)
        4. Updates theta
            -> theta = theta - gradient - penalization
        5. Computes gradient for theta_zero
            -> -(alpha / m) * SUM[(y_true / y_pred) - ((1 - y_true) / (1 - y_pred))]
        6. Updates theta_zero
            -> theta_zero = theta_zero - gradient_zero
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        m: int
            The number of examples in the dataset
        """
        # predicted y (uses the sigmoid function)
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        # compute the gradient vector (of theta) given a learning rate alpha
        # vector of shape (n_features,) -> gradient[k] updates self.theta[k]
        grad_prod_1 = np.dot(dataset.y / y_pred, dataset.X)
        grad_prod_2 = np.dot((1 - dataset.y) / (1 - y_pred), dataset.X)
        gradient = -(self.alpha / m) * (grad_prod_1 - grad_prod_2)
        # compute the penalization term
        penalization_term = self.theta * self.alpha * (self.l2_penalty / m)
        # update theta
        self.theta = self.theta - gradient - penalization_term
        # compute theta_zero gradient (penalization term is 0)
        gradient_zero = -(self.alpha / m) * np.sum((dataset.y / y_pred) - ((1 - dataset.y) / (1 - y_pred)))
        # update theta_zero
        self.theta_zero = self.theta_zero - gradient_zero

    def _regular_fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset. Does not update the learning rate (self.alpha). Covergence is attained
        whenever the difference of cost function values between iterations is less than <self.tolerance>.
        Returns self (fitted model).
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        """
        # get the shape of the dataset
        m, n = dataset.shape()
        # initialize the model parameters (it can be initialized randomly using a range of values)
        self.theta = np.zeros(n)
        self.theta_zero = 0
        # main loop -> gradient descent
        i = 0
        converged = False
        while i < self.max_iter and not converged:
            # compute gradient descent iteration (update model parameters)
            self._gradient_descent_iter(dataset, m)
            # add new entry to self.cost_history
            self.cost_history[i] = self.cost(dataset)
            # verify convergence
            converged = abs(self.cost_history[i] - self.cost_history.get(i-1, np.inf)) < self.tolerance
            i += 1
        return self

    def _adaptative_fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset. Updates the learning rate (self.alpha) by halving it every
        time the difference of cost function values between iterations is less than <self.tolerance>.
        Returns self (fitted model).
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        """
        # get the shape of the dataset
        m, n = dataset.shape()
        # initialize the model parameters (it can be initialized randomly using a range of values)
        self.theta = np.zeros(n)
        self.theta_zero = 0
        # main loop -> gradient descent
        for i in range(self.max_iter):
            # compute gradient descent iteration (update model parameters)
            self._gradient_descent_iter(dataset, m)
            # add new entry to self.cost_history
            self.cost_history[i] = self.cost(dataset)
            # update learning rate
            is_lower = abs(self.cost_history[i] - self.cost_history.get(i-1, np.inf)) < self.tolerance
            if is_lower: self.alpha /= 2
        return self

    def fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset. If self.adaptative_alpha is True, fits the model by updating the
        learning rate (alpha). Returns self (fitted model).
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to fit the model to)
        """
        self.fitted = True
        return self._adaptative_fit(dataset) if self.adaptative_alpha else self._regular_fit(dataset)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the output of the dataset. Applies the sigmoid function, returning 0 if
        y_pred < 0.5, otherwise 1.
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to predict the output of)
        """
        if not self.fitted:
            raise Warning("Fit 'LogisticRegression' before calling 'predict'.")
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        mask = y_pred >= 0.5
        y_pred[~mask], y_pred[mask] = 0, 1
        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Computes and returns the accuracy score of the model on the dataset.
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the accuracy on)
        """
        if not self.fitted:
            raise Warning("Fit 'LogisticRegression' before calling 'score'.")
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Computes and returns the value of the cost function (J function) of the model on the dataset
        using L2 regularization.
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the cost function on)
        """
        if not self.fitted:
            raise Warning("Fit 'LogisticRegression' before calling 'cost'.")
        m = dataset.shape()[0]
        # compute actual predictions (not 'binarized')
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        # calculate value of the cost function
        error = -np.sum(dataset.y * np.log(predictions) + (1-dataset.y) * np.log(1-predictions)) / m
        regularization = (self.l2_penalty / (2*m)) * np.sum(self.theta**2)
        return error + regularization