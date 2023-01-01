import numpy as np


class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        :param input_size: int, the number of inputs the layer will receive
        :param output_size: int, the number of outputs the layer will produce
        """
        # Parameters
        self.input_size = input_size
        self.output_size = output_size
        # Attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward (self, X: np.ndarray) -> np.ndarray:
        """
        It performs forward pass of the layer using the given input.
        Returns a 2 dimension numpy array with shape (1, output_size).
        :param X: np.ndarray, the input to the layer
        :return: np.ndarray, the output to the layer.
        """
        return np.dot(X, self.weights) + self.bias
        # dot to multiply the matrix

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        It computes the backward pass of the layer. Returns the error of the previous layer.
        :param error: np.ndarray, error value of the loss function
        :param learning_rate: float, the learning rate
        :return: np.ndarray, error of the previous layer
        """
        #get error of the previous layer
        error_to_propagate = np.dot(error, self.weights.T)

        #update the weights and bias
        self.weights = self.weights - learning_rate * np.dot(self.X.T, error)
        # x.T is used to multiply the error by the input data due to matrix multiplication rules
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)
        return error_to_propagate

class SigmoidActivation:
    """
     A sigmoid activation layer.
    """
    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        # Attribute
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        It performs a forward pass of the layer using the given input.
        Returns a 2 dimension numpy array with shape (1, output_size).
        :param X: np.ndarray, input to the layer
        :return: np.ndarray, output to the layer
        """
        self.X = X
        return 1/ (1+ np.exp(-self.X)) # sigmoid function

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        It performs a backward pass of the layer. Returns the error of the previous layer.
        :param error: np.ndarray, error value of the loss function
        :return: np.ndarray, error of the previous layer
        """
        sigmoid_derivative = (1 / (1+ np.exp(-self.X))) * (1 - (1 / (1 + np.exp(- self.X))))
        error_to_propagate = error * sigmoid_derivative
        return error_to_propagate

# Exercise 10 - 10.1. Add a new layer SoftMaxActivation

class SoftMaxActivation:
    """
    A SoftMax activation layer.
    """
    def __init__(self):
        """
        Initialize the SoftMax activation layer.
        """
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        It performs a forward pass of the layer using the given input.
        Returns a 2 dimension numpy array with shape (1, output_size).
        :param X: np.ndarray, input to the layer
        :return: np.ndarray, output to the layer
        """
        # exponential of vetor zi (ezi)
        ezi = np.exp(X - np.max(X))
        # formula: ezi / sum of ezi
        formula = ezi / np.sum(ezi, axis=1, keepdims=True)
        # axis=1 means the sum happens by row
        # keepdims=True means the dimension of the array is kept
        return formula

# Exercise 10 - 10.2. Add a new layer ReLUActivation
class ReLUActivation:
    """
    A rectified linear (ReLu) activation layer.
    """
    def __init__(self):
        """
        Initialize the ReLu activation layer.
        """
        self.X = None
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        It performs a forward pass of the layer using the given input.
        Returns a 2 dimension numpy array with shape (1, output_size).
        :param X: np.ndarray, input to the layer
        :return: np.ndarray, output to the layer
        """
        self.X = X
        # formula: maximum between between 0 and X, using 0 as minimum, so we don't have negative values
        formula_relu = np.maximum(0, self.X)
        return formula_relu

# Exercise 12 - 12.1.
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        It performs a backward pass of the layer. Returns the error of the previous layer.
        :param error: np.ndarray, error value of the loss function
        :return: np.ndarray, error of the previous layer
        """
        # replace all positive elements (>0) to 1 and set all negative elements (<0) to 0 - using np.where
        relu_derivative = np.where(self.X > 0, 1, 0)
        # multiplication between error and previous values
        error_to_propagate = error * relu_derivative
        return error_to_propagate

# For Exercise 10.5. - Add new layer LinearActivation
class LinearActivation:
    """
    A linear activation layer.
    """
    def __init__(self):
        """
        Initialize the linear activation layer.
        """
        pass

    def forward(input_data: np.ndarray) -> np.ndarray:
        """
        It computes the linear relationship.
        :param X: np.ndarray, input to the layer
        :return: np.ndarray, the linear relationship
        """
        return input_data






