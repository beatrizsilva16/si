import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.
    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.
    Attributes
    ----------
    """
    def __init__(self, models: list, weighted: bool = False):
        """
        Initialize the ensemble classifier.
        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        """
        # parameters
        self.models = models
        self.weighted = weighted

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the models according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The training data.
        Returns
        -------
        self : VotingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)

        return self

    def _get_majority_vote(self, predictions: np.ndarray) -> int:
        """
        Helper function which determines and returns the most common label in a set
        of predictions.
        Parameters
        ----------
        predictions: np.ndarray
            An array consisting of the labels predicted for a given example
        """
        labels, counts = np.unique(predictions, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts and returns the output of the dataset. To do so, it uses voting to
        combine the predictions of the models in <self.models>. If <self.weighted> is
        set to True, model predictions are weighted according to the respective scores.
        Note: assumes that all models use the same scoring metric.
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset containing the examples to be labeled)
        """
        # array containing the outputs of each k models in k rows
        predictions = np.array([model.predict(dataset) for model in self.models])
        # weigh model predictions based on the respective scores
        if self.weighted:
            # get model scores
            scores = [model.score(dataset) for model in self.models]
            # scale scores so that min_score = 1
            min_scr = min(scores)
            scores_sc = np.array([round((1 / min_scr) * scr) for scr in scores])
            # update predictions in order to account for the computed weights
            predictions = np.repeat(predictions, repeats=scores_sc, axis=0)
        # voting is performed col-wise so that outputs of different models are compared
        return np.apply_along_axis(self._get_majority_vote, axis=0, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Computes and returns the accuracy score of the model on the dataset.
        Parameters
        ----------
        dataset: Dataset
            A Dataset object (the dataset to compute the accuracy on)
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
