import numpy as np
from si.data.dataset import Dataset

import sys
PATHS = ["../data"]
sys.path.extend(PATHS)


class PCA:
    """
        PCA implementation to reduce the dimensions of a given dataset. It uses SVD (Singular
        Value Decomposition) to do so.
        """

    def __init__(self, n_components: int = 10):

        """
        PCA implementation to reduce the dimensions of a given dataset. It uses SVD (Singular
        Value Decomposition) to do so.
        Parameters
        ----------
        n_components: int (default=10)
        The number of principal components to be computed
        Attributes
        ----------
        fitted: bool
        Whether 'PCA' is already fitted
        mean: np.ndarray
        The mean value of each feature of the dataset
        components: np.ndarray
        The first <n_components> principal components
        explained_variance: np.ndarray
        The variances explained by the first <n_components> principal components
        """
        if n_components < 1:
            raise ValueError('The value of n_components must be greater than 0.')

        self.n_components = n_components
        self.fitted = False
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> "PCA":
        """
        Fits PCA by computing the mean value of each feature of the dataset, the first
        <n_components> principal components and the corresponding explained variances.
        Returns self.
        Parameters: dataset - Dataset
                     A Dataset object
        """

       # get center data
        self.mean = np.mean(dataset.X, axis=0) #inferir média das amostras
        data_centered = dataset.X - self.mean # subtraí a média ao dataset (X -mean)

        # get SVD
        U, S, V_T = np.linalg.svd(data_centered, full_matrices=False)

        # get principal components
        self.components = V_T[:self.n_components]

        # get explained variance
        n = dataset.shape()[0]
        ev_formula = (S ** 2) / (n-1)
        self.explained_variance = ev_formula [:self.n_components]
        self.fitted = True
        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
            Transforms the dataset by reducing X (X_reduced = X * V, V = self.components.T).
            Returns X reduced.

            Parameteres: dataset - Dataset
                            A Dataset object
        """

        if not self.fitted:
            raise Warning('Fit pca before calling transform.')

        # get center data
        data_centered = dataset.X - self.mean

        # get x reduced
        return np.dot(data_centered, self.components.T)

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
            Fits PCA and transforms the dataset by reducing X. Returns X reduced.

            Parameters
            ----------
            dataset: Dataset
            A Dataset object
        """

        self.fit(dataset)
        return self.transform(dataset)







