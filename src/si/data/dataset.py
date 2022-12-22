import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, features: list, label: str):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        # dimensões dataset
        return self.X.shape

    def has_label(self):
        # verfica se o dataset tem y
        return self.y is not None
        # devolve True ou False

    def get_classes(self):
        # devolve as classes do dataset (valores possíveis de y)
        return np.unique(self.y)

    def get_mean(self):
        # devolve média para cada variável dependente
        return np.mean(self.X, axis=0)  # axis 0: refers to horizontal axis or rows, axis 1: refers to vertical axis or
        # columns (exemplos)

    def get_variance(self):
        # devolve variância para cada variável dependente
        return np.var(self.X, axis=0)

    def get_median(self):
        # devolve mediana para cada variável dependente
        return np.median(self.X, axis=0)

    def get_min(self):
        # devolve valor mínimo para cada variável dependente
        return np.min(self.X, axis=0)

    def get_max(self):
        # devolve valor máximo para cada variável dependente
        return np.max(self.X, axis=0)

    def summary(self):
        # devolve pandas DataFrame com todas as métricas descritivas
        df = pd.DataFrame(columns=self.features)
        df.loc['mean'] = self.get_mean()
        df.loc['variance'] = self.get_variance()
        df.loc['median'] = self.get_median()
        df.loc['min'] = self.get_min()
        df.loc['max'] = self.get_max()
        return df


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [1, 2, 3]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(x, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
