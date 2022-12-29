from typing import Tuple, Sequence
import pandas as pd
import numpy as np


class Dataset:
    """
          Dataset represents a machine learning tabular dataset.
          Parameters:
          X: numpy.ndarray (n_samples, n_features)
              The feature matrix
          y: numpy.ndarray (n_samples, 1)
              The label vector
          features: list of str (n_features)
              The feature names
          label: str (1)
              The label name
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):

        if X is None:
            raise ValueError('X cannot be None')

        if features is None:
            features = [str(i) for i in range(X.shape[1])]

        else:
            features = list(features)

        if y is not None and label is None:
            label = 'y'

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        # dimensões do dataset, returns : tuple(n_samples, n_features)
        return self.X.shape

    def has_label(self) -> bool:
        # verfica se o dataset tem y
        return self.y is not None
        # devolve True ou False

    def get_classes(self) -> np.ndarray:
        # devolve as classes do dataset (valores possíveis de y)

        if self.y is None:
            raise ValueError('Dataset does not have a label')
        return np.unique(self.y)

    def get_mean(self) -> np.ndarray:
        # devolve média para cada variável dependente
        return np.nanmean(self.X, axis=0)
        # axis 0: refers to horizontal axis or rows, axis 1: refers to vertical axis or
        # columns (exemplos)

    def get_variance(self) -> np.ndarray:
        # devolve variância para cada variável dependente
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        # devolve mediana para cada variável dependente
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        # devolve valor mínimo para cada variável dependente
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        # devolve valor máximo para cada variável dependente
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
               Returns a pd.DataFrame containing some descriptive metrics (mean, variance, median,
               minimum value and maximum value) of each feature.
        """

        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }

        return pd.DataFrame.from_dict(data, orient='index', columns=self.features)

    def dropna(self):
        """
        Removes all the samples that have at least a "null" value (NaN).
        """

        indexList = [np.any(i) for i in np.isnan(self.X)]
        self.X = np.delete(self.X, indexList, axis=0)
        if self.y is not None:
            self.y = np.delete(self.y, indexList, axis=0)

    def fillna(self, value: float):
        """
            Replaces "null" values (NaN) by another value given by the user.

            Parameters
            ----------
            value: float
                the value that will replace "null" values (NaN)
        """
        self.X = np.nan_to_num(self.X, nan=value)

    """
        Class method is a method that is bound to a class rather than its object.
        It doesn't require creation of a class instance
        """
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
                Creates a Dataset object from a pandas DataFrame
                Parameters
                ----------
                df: pandas.DataFrame
                    The DataFrame
                label: str
                    The label name
                Returns
                -------
                Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()

        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
            Converts the dataset to a pandas DataFrame
            Returns
            -------
            pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data
        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)
