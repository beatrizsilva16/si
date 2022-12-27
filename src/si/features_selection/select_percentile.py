import numpy as np
from si.data.dataset import Dataset
from statistics.f_classification import f_classification
from typing import Callable

class SelectPercentile():
    '''
    Select features according to a percentile of the highest scores
    parameters: score_func: callable, default=f_classif
                    Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues)
                    or a single array with scores. Default is f_classif (see below “See Also”).
                    The default function only works with classification tasks.

                percentile: int, default=10
                    Percent of features to keep.
    '''
    def __init__(self, score_func: Callable = f_classification, percentile: float):

        """
        Select features according to a percentile chosen by a user
        :param score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        :param percentile: float
             Percentile to select the features.
        """

        if percentile < 0 or percentile > 1:
            raise ValueError('The value of 'percentile' must be in [0,1].')
        self.score_func = score_func
        self.percentile = percentile
        self.F_value = None
        self.p_value = None

    def fit(self, dataset: Dataset):
        """
        Fits SelectPercentile by computing the F-scores and p-values of the dataset's features.
        Returns self.
        """

        self.F_value, self.p_value = self.score_func(dataset) # retorna os valores F e p
        return self

    def transform(self, dataset: Dataset):
        k = round(len(dataset.features) * self.percentile * 0.01)
        idxs = np.argsort(self.F)[-k]
        fetures = np.array(dataset.features)[idxs]
        return Dataset (X = dataset.X[:, idxs], y=dataset.y, features = list(features), label=dataset.label)


    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)
