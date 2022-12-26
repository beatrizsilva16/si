import numpt as np

from si.data.dataset import Dataset
from si.metrics.mse import mse

class RigdeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    : param:  l2_penalty: float
        The L2 regularization parameter
              alpha: float
        The learning rate
              max_iter: int
        The maximum number of iterations

    : atributes:  theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
                  theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """

    def __init__(self, 12_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):

        #Parameters
        self.12_penalty = 12_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        #attributes
        self.theta = None
        self.theta_zero = None

    def fit(self, dataset: Dataset) -> 'RigdeRegression':
        """
        Fits the model to the dataset
        :param: dataset
        :return: self: RigdeRegression
            The fitted model
        """
        m,n = dataset.shape()

        #initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zeros = 0

        # gradient descent
        for i in range(self.max_iter):
            #predicted y
            y_pred = np.dot(dataset.x, self.theta) + self.theta_zero

            #computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1/m)) * np.dot(y_pred - dataset.y, dataset.x)

            #computing the penalty
            penalization_term = self.alpha * (self.12_penalty / m )* self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1/m)) * np.sum(y_pred - dataset.y)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset
        :param dataset: Dataset
        :return: predictions: np.array
            The predictions of the dataset
        """
        return np.dot(dataset.x, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset
        :param dataset:
            The dataset to compute the MSE on
        :return: mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        :param dataset:Dataset
            The dataset to compute the cost function on
        :return:cost: float
            The cost function of the model
        """

        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + self.12_penalty)










