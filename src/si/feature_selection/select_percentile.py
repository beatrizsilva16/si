import numpy as np
from si.data.dataset import Dataset
import sys
PATHS = ["../data", "../statistics"]
sys.path.extend(PATHS)


from si.statistics.f_classification import f_classification
#Exercício 3 (3.1. e 3.2.)


class SelectPercentile:
    '''
    Select features according to a percentile of the highest scores
    parameters: score_func: callable, default=f_classif
                    Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues)
                    or a single array with scores. Default is f_classif (see below “See Also”).
                    The default function only works with classification tasks.
                percentile: int, default=10
                    Percent of features to keep.
    '''
    def __init__(self, dataset, percentile) -> None:
        """
        Select features according to a percentile chosen by a user
        :param score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        :param percentile: float
             Percentile to select the features.
        """

        if percentile < 0 or percentile > 1:
            raise ValueError('The value of percentile ust be in [0,1]')

        self.dataset = dataset
        self.score_func = f_classification(dataset)
        self.percentile =int(percentile*len(dataset.features))
        self.F = None
        self.p = None

    def fit(self):
        """
        Fits SelectPercentile by computing the F-scores and p-values of the dataset's features.
        Returns self.
        """

        self.F, self.p = self.score_func # retorna os valores F e p
        return self

    def transform(self):
        idxs = np.argsort(self.F)[-self.percentile]
        features = np.array(self.dataset.features)[idxs]
        return Dataset(self.dataset.X[:, idxs], self.dataset.y, list(features), self.dataset.label)

    def fit_transform(self):
        self.fit()
        return self.transform()