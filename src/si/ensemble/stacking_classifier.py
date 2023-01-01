import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.neighbors.knn_classifier import KNNClassifier


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

    def __init__(self, models: list, final_model=KNNClassifier, weighted: bool = False):
        """
        Initialize the ensemble classifier.
        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        """
        # parameters
        self.models = models
        self.final = final_model
        self.weighted = weighted
        # attributes
        self.scores_sc = None

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
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
        #fit the ensemble on training data -1
        for model in self.models:
            model.fit(dataset)

        #get the predictions of each model for traning data - 2
        predictions = np.array([model.predict(dataset) for model in self.models])

        # weigh model predictions based on the respective scores -3
        if self.weighted:
            # get model scores
            scores = [model.score(dataset) for model in self.models]
            # scale scores so that min_score = 1 (store scores_sc -> use in 'predict')
            min_scr = min(scores)
            self.scores_sc = np.array([round((1/min_scr)*scr) for scr in scores])
            # update predictions in order to account for the computed weights
            predictions = np.repeat(predictions, repeats=self.scores_sc, axis=0)

        # create a Dataset object containing the predictions
        ds_train = Dataset(predictions.T, dataset.y)
        # fit the final model based on the predictions of the ensemble -4
        self.final_model.fit(ds_train)
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
