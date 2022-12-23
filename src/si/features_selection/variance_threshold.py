from src.si.data.dataset import Dataset

class VarianceThreshold:

    """
     Parameters
    ----------
    threshold : float, default=0
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.
    """

    def __int__(self, threshlod: float):

        if threshlod < 0:
            raise ValueError("Threshold must be positive")

        self.threshold = threshlod
        self.variance = None #está vazio porque não vamos calcular variance aqui, não há dataset



    def fit(self, dataset: object):

        """ Estima/calcula a variância de cada feature; retorna o self (ele próprio)
         Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.
        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        Returns
        -------
        self : object
            Returns the instance itself."""

        self.variance = dataset.get_var()
        return self # o fit returna ele próprio



    def transform (self, dataset: object) -> object: # vai retornar um novo dataset

        """
        Seleciona todas as features com variância superior ao threshold e
        retorna o X selecionado
        """
        mask = self.variance > self.threshold
        newX = dataset.X[:,mask]
        features = np.array(dataset.features)[mask] # seleciona as features com valor de threshold superior ao de variance
        return Dataset(newX, y=dataset.y, features=list(features),label=dataset.label)



    def fit_transform(self, dataset:object) -> object:

        """
        Corre o fit e depois o transform
        """

        model = self.fit(dataset)
        return model.transform(dataset)


if __name__ == '__main__':

    import numpy as np

    dataset = Dataset (np.array([[0, 2, 0, 1]
                                   [0, 1, 4, 3]
                                   [0, 1, 1, 3]
                                   [0, 4, 0, 2]]),
                      np.array([1,2,3,4]),
                      ["1","2","3","4"], "5")

    ex1 = VarianceThreshold(1)
    print(ex1.fit_transform(dataset))