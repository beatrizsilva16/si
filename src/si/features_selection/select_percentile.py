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
    def __init__(self, score_func, percentile: int):
        self.score_func = score_func
        self.percentile = percentile
        self.F_value = None
        self.p_value = None

    def fit(self, dataset) -> self:
        self.f_value, self.p_value = self.score_func(dataset) # retorna os valores F e p
        return self

    def transform(self, dataset):
        value = len(dataset.features) #valor total de features no dataset
        mask = value * (percentile/100) #percentil dado em percentagem
        idxs = np.argsort(self.F_value)[- mask:]
        features = np.array(dataset.features)[idxs]  # vai selecionar as features utilizando os indexs
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
