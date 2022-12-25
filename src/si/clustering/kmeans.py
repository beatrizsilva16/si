from typing import Callable

import numpy as np
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance

class KMeans:
    """
    Identify clusters of data objects in a dataset
    Parameters:
        k – number of clusters/centroids
        max_iter – number maximum of interaction. Inteiro
        distance – function that calculates the distance

    Estimated parameters:
        centroids – mean of samples in each centroid
        labels – vector with a label in each centroid
    """

    def __init__(self, k: int, max_iter: int = 1000, distance : Callable = euclidean_distance):
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.centroids = None
        self.labels = None

    def __init__centroids(self, dataset: Dataset):
        """
        Initializes the centroids

        Parameters: dataset - Dataset object
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]#randomly chooses k samples from the dataset
        self.centroids = dataset.x[seeds] #use the k samples as centroids

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Gets the index of the closest centroid for a given sample
        parameters sample: np.ndarray. shape(n_features,). One sample

        return: ndarray of centroid with the shortest distance of each point
        """
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis=0) #0 porque só temos um vetor em linhas e vai buscar o index da menor distância
        return closest_centroid_index

    def fit(self, dataset: Dataset) -> 'KMeans':
        """
        Method for calculate the distance between one sample and centroids from dataset
        np.random.permutation : creation a random vector

        Parameters dataset: Dataset Object
        return: SelectKBest object
        """

        self. __init__centroids(dataset)
        convergence = False # tells if the algorithm has converged
        i = 0
        labels = np.zeros(dataset.shape()[0])

        while not convergence and i < self.max_iter:

            #get closest centroids
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis = 1, arr=dataset.x)

            #compute the new centroids
            centroids = []
            for j in range(self.k):
                centroid = np.mean(dataset.x[new_labels == j], axis=0)
                centroids.append(centroid)

            self.centroids = np.array(centroids)

            # check if the centroid have changed
            convergence = np.any(new_labels !=labels)

            #replace labels
            labels = new_labels

            #increment counting
            i += 1

        self.labels = labels
        return self

    def _get_distance(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the distance between two samples

        Parameter x: Sample
        Return distances between each sample and the closest centroid
        """
        return self.distance(x, self.centroids)

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Selects the k best features

        Parameter dataset: Dataset object
        Return: Transformed dataset
        """
        centroids_distances = np.apply_along_axis(self._get_distance, axis=1, arr=dataset.x)
        return centroids_distances

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the label of a given sample
        Parameter dataset: Dataset object
        Return: Predicted labels
        """
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.x)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        """
        Fits and predicts the labels of the dataset
        Parameters:
        """

        self.fit(dataset)
        return self.predict(dataset)








