from ctypes import Union
import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accurancy
from si.statistics.euclidean_distance import euclidean_distance

class KNNClassifier:
    def __init__(self, k:int, distance: euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset):
        """
        Method that stores the dataset
        :param dataset: Dataset object
        :return: dataset
        """
        self.dataset = dataset
        return self

    def _get_closet_label(self, x:np.array) -> Union[int, str]:
        """
        Returns the closest label of the given sample
        """

        # Compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.x)

        # Get the k nearest neighors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # Get the most common label
        labels, counts = np.unique(k_nearest_neighbors_labels,return_counts = True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the classes of the guven dataset

        """
        return np_along_axis(self._get_closet_label, axis=1, arr=dataset.x)

    def score(self, dataset: Dataset) -> float:
        """
        Returns de accuracy of the model
        :param dataset: Dataset de teste (y_true) (y correspondem às labels)
        :return: Valor de accuracy do modelo
        """

        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)