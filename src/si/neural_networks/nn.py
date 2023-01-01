import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse, mse_derivative


class NN:
    """
    The NN is the Neural Network model.
    It comprehends the model topology including several neural network layers.
    The algorithm for fitting the model is based on backpropagation.
    """
    def __init__(self, layers: list, epochs: int = 1000, learning_rate: float = 0.01, loss: Callable = mse,
                 loss_derivative: Callable = mse_derivative, verbose: bool = False):
        """
        Initializes the neural network model.
        :param layers: list, list of layers in the neural network
        :param epoch: int, number of epochs to train the model
        :param learning_rate: float, the learning rate of the model
        :param loss: Callable, the loss function to use
        :param loss_derivative: Callable, the derivative of loss function to use
        :param verbose: bool, whether to print the loss at each epoch
        """
        # Parameters
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose
        # Attributes
        self.history = {}

    def fit(self, dataset: Dataset) -> 'NN':
        X = dataset.X
        y = dataset.y

        for epoch in range(1, self.epochs +1):
            #forward propagation
            y_pred = np.array(dataset.X)
            y_true = np.reshape(dataset.y, (-1, 1))

            for layer in self.layers:
                y_true = layer.forward(X)

            #backward propagation
            error = self.loss_derivative(y_true, y_pred)
            for layer in self.layers[::-1]: # last layer
                error = layer.backward(error, self.learning_rate)

            #save history
            cost = self.loss(y_true, y_pred)
            self.history[epoch] = cost

            #print loss - loss function
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost{cost}')
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the output of the given dataset.
        :param dataset: Dataset, dataset to predict the output of
        :return: np.ndarray, predicted output
        """
        X = dataset.X

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost of the model on the given dataset.
        :param dataset: Dataset, dataset to compute the cost on
        :return: float, the cost of the model
        """
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
        It computes the score of the model on the given dataset.
        :param dataset: Dataset, dataset to compute the score on
        :param scoring_func: Callable, the scoring function to use
        :return: float, the score of the model
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)

    if __name__ == '__main__':
        pass


