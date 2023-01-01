import numpy as np

from si.data.dataset1 import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.
    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.
    Attributes
    ----------
    """

    def __init__(self, models, final):
        """
        Initialize the ensemble classifier.
        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        """
        # parameters
        self.models = models
        self.final = final

    def fit(self, dataset: Dataset, final_model=None) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The training data.
        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)

        fit = np.array([model.predict(dataset) for model in self.models]).transpose()
        prev_prd_data = Dataset(X=fit, y=dataset.y)
        self.final.fit(prev_prd_data)

        return self

    def predict(self, dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        dataset : Dataset
            The test data.
        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        prev_prd_data = Dataset(X=predictions, y=dataset.y)
        return self.final.predict(prev_prd_data)

    def score(self, dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.
        Parameters
        ----------
        dataset : Dataset
            The test data.
        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))