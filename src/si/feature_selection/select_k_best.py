import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest:

    """
    Select features according to the k highest scores.
    Parameters: score_func: callable, default=f_classif
                kint or “all”, default=10

    """
    def __init__(self, score_func: Callable = f_classification, k: int = 10):

        if k < 1:
            raise ValueError('The value of k must be greater than 0.')
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
               It fits SelectKBest to compute the F scores and p-values.
               Parameters
               ----------
               dataset: Dataset
                   A labeled dataset
               Returns
               -------
               self: object
                   Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
               It transforms the dataset by selecting the k highest scoring features.
               Parameters
               ----------
               dataset: Dataset
                   A labeled dataset
               Returns
               -------
               dataset: Dataset
                   A labeled dataset with the k highest scoring features.
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]  # selecionar as features com base nos idx
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
               It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.
               Parameters
               ----------
               dataset: Dataset
                   A labeled dataset
               Returns
               -------
               dataset: Dataset
                   A labeled dataset with the k highest scoring features.
        """
        self.fit(dataset)
        return self.transform(dataset)
