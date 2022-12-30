from ctypes import Union
import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
import sys
PATHS = ["src/si/data/dataset", "src/si/statistics/euclidean_distance", "src/si/metrics/accuracy/accuracy"]
sys.path.append(PATHS)


class KNNClassifier:
    """
        KNN Classifier
        The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
        a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
        looking at the classes of the k-nearest samples in the training data.
        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        Attributes
        ----------
        dataset: np.ndarray
            The training data
    """
    def __init__(self, k: int, distance: euclidean_distance):
        """
                Initialize the KNN classifier
                Parameters
                ----------
                k: int
                    The number of nearest neighbors to use
                distance: Callable
                    The distance function to use
        """

        if k < 1:
            raise ValueError('The value of k must be greater than 0.')
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

    def _get_closet_label(self, sample: np.array) -> Union[int, str]:
        """
                Predicts the class with the highest frequency.
                :param x: Sample.
                :return: Indexes of the classes with the highest frequency.
        """

        # Compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.x)

        # Sort the distances and get indexes
        knn = np.argsort(distances)[:self.k]  #get the first k indexes of the sorted distances array

        # Get the k nearest neighors
        knn_labels = self.dataset.y[knn]

        # Get the most common label
        labels, counts = np.unique(knn_labels, return_counts=True)

        # Gets the most frequent class
        high_freq_lab = labels[np.argmax(counts)] # ?? get the indexes of the classes with the highest frequency/count
        return high_freq_lab

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
                It predicts the classes of the given dataset
                Parameters
                ----------
                dataset: Dataset
                    The dataset to predict the classes of
                Returns
                -------
                predictions: np.ndarray
                    The predictions of the model
        """
        return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.x)

    def score(self, dataset: Dataset) -> float:
        """
                It predicts the classes of the given dataset
                Parameters
                ----------
                dataset: Dataset
                    The dataset to predict the classes of
                Returns
                -------
                predictions: np.ndarray
                    The predictions of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred) # Returns the number of correct predictions divided
        # by the total number of predictions (accuracy)
        # The correct predictions are calculated by the predictions and the true values from the dataset