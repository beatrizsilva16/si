import numpy as np
from data.dataset import Dataset
from statistics.f_classification import f_classification

class SelectPercentile():
    '''
    Select features according to a percentile of the highest scores
    parameters: score_funccallable, default=f_classif
                Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues)
                or a single array with scores. Default is f_classif (see below “See Also”).
                The default function only works with classification tasks.

                percentileint, default=10
                Percent of features to keep.
    '''
    def __init__(self, score_func, percentile: int):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset) -> self:
        self.f_value, self.p_value = self.score_func(dataset)
        return self

    def transform(self, dataset):
        value = len(dataset.features) #valor total de features no dataset
        mask = value * (percentile/100) #percentil dado em percentagem
        idxs = np.argsort(self.F)[- mask:]
        features = np.array(dataset.features)[idxs]  # vai selecionar as features utilizando os indxs
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    if __name__ == '__main__':
        dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                      [0, 1, 4, 3],
                                      [0, 1, 1, 3]]),
                          y=np.array([0, 1, 0]),
                          features=["f1", "f2", "f3", "f4"],
                          label="y")
        select = SelectPercentile(f_classification,25)
        # chamar o método f_classification para o cálculo e introduzir valor de k
        select = select.fit(dataset)
        dataset = select.transform(dataset)
        print(dataset)