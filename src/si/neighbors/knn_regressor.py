import numpy as np
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance
from typing import Callable


class KNNRegressor:
    """
    Implements the K-Nearest Neighbors regressor. Distances between test examples and examples
    present in the training data can be computed using one of two distinct formulas:
        - euclidean_distance: sqrt(SUM[(pi - qi)^2])
        - manhattan_distance: SUM[abs(pi - qi)]
    """

    def __init__(self, k: int = 4, weighted: bool = False, distance: Callable = euclidean_distance):
        """
        Implements the K-Nearest Neighbors regressor. Distances between test examples and examples
        present in the training data can be computed using one of two distinct formulas:
            - euclidean_distance: sqrt(SUM[(pi - qi)^2])
            - manhattan_distance: SUM[abs(pi - qi)]
        Parameters
        ----------
        k: int (default=4)
            Number of neighbors to be used
        weighted: bool (default=False)
            Whether to weight closest neighbors when predicting labels
        distance: callable (default=euclidean_distance)
            Function used to compute the distances
        Attributes
        ----------
        fitted: bool
            Whether the model is already fitted
        weights_vector: np.ndarray
            The weights to give to each closest neighbor when predicting labels (only applicable
            when 'weights' is True)
        dataset: Dataset
            A Dataset object (training data)
        """
        # parameters
        if k < 1:
            raise ValueError("The value of 'k' must be greater than 0.")
        self.k = k
        self.weighted = weighted
        self.distance = distance
        # attributes
        self.fitted = False
        if self.weighted:
            self.weights_vector = np.arange(self.k, 0, -1)
        self.dataset = None

    def fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Stores the training dataset. Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (training data)
        """
        self.dataset = dataset
        self.fitted = True
        return self

    def _get_closest_labels_mean(self, sample: np.ndarray) -> float:
        """
        Returns the predicted label of the sample given as input. The label is determined by
        computing the mean value of the labels of the sample's <self.k> closest neighbors.
        Parameters
        ----------
        sample: np.ndarray
            The sample to be labeled
        """
        # calculate distances
        distances = self.distance(sample, self.dataset.X)
        # determine indices of the closest neighbors
        label_indices = np.argsort(distances)[:self.k]
        # get the values at the previous indices
        label_vals = self.dataset.y[label_indices]
        # tranform labels vector to account for weights (if applicable)
        if self.weighted:
            label_vals = np.repeat(label_vals, self.weights_vector)
        # compute the mean value and return it
        return np.mean(label_vals)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the labels of the dataset given as input.
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (testing data)
        """
        if not self.fitted:
            raise Warning("Fit 'KNNRegressor' before calling 'predict'.")
        return np.apply_along_axis(self._get_closest_labels_mean, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Calculates and returns the error between the predicted and true classes. To compute
        the error, it uses the RMSE: sqrt((SUM[(ti - pi)^2]) / N).

        Parameters
        ----------
        dataset: Dataset
            A Dataset object (testing data)
        """
        if not self.fitted:
            raise Warning("Fit 'KNNRegressor' before calling 'score'.")
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)

