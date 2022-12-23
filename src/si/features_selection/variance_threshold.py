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

    def __int__(self, threshlod:float):
        self.threshold = threshlod
        self.variance = None #está vazio porque não vamos calcular variance aqui



    def fit(self, dataset: object):

        """Learn empirical variances from X.
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

        mask = self.variance > self.thresh
        newX = dataset.X[:,mask]


        if not (dataset.features is None):
            dataset.features = [elem for ix,elem in enumerate(dataset.features) if mask[ix]]

            print (dataset.features)

        return Dataset(newX, dataset.y, dataset.features, dataset.label)



    def fit_transform(self, dataset:object) -> object:
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

    temp = VarianceThreshold(1)
    print(temp.fit_transform(dataset))