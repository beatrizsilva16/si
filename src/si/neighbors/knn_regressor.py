import numpy as np

from typing import Union
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:
    """
        KNN Regressor - Implements the K-Nearest Neighbors regressor on a ML model  based on a similarity measure
        (like euclidean distance).
    """
    def __init__(self, k: int = 1, distance: callable = euclidean_distance):
        """
        Initialize the KNN Regressor.
            :param k: int, number of nearest neighbors to be used
            :param distance: callable, distance function to use
        """
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset):
        """
        Stores the dataset.
        :param dataset: Dataset object
        :return: The dataset
        """
        self.dataset = dataset
        return self

    def _get_closet_label(self, sample: np.ndarray) -> np.ndarray:
        """
        Calculates the class with the highest frequency.
        :param x: Array of samples.
        :return: Indexes of the classes with the highest frequency
        """

        # Calculates the distance between the samples and the dataset
        distances = self.distance(sample, self.dataset.X)

        # Sort the distances and get indexes
        label_indexs = np.argsort(distances)[:self.k]

        # Get the labels values of indexes obtained
        labels_values = self.dataset.y[label_indexs]

        # Compute the mean value and return it
        return np.mean(labels_values)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class with the highest frequency
        :return: Class with the highest frequency.
        """
        return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.
        :return: Accuracy of the model.
        """
        predictions = self.predict(dataset)

        return rmse(dataset.y, predictions)
