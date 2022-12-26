import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accurancy

class VotingClassifier:
    def __int__(self, models: list):
        """

        :param models: Lista de dados
        :return:
        """
        self.models = models

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the model to the dataset
        :param dataset:Dataset object to fit the model to
        :return:VotingClassifier
        """
        for model in self.models:
            model.fit(dataset)

    def predict(self, dataset) -> np.array:
        """
        Combines the previsions of each model with a voting system.
        :param dataset:Dataset object to predict the labels of.
        :return:the most represented class
        """
        def _get_most_represented_class(pred: np.array) -> int:
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]

        #list of the predictions
        predictions = []

        for model in self.models:
            predictions.append(model.predict(dataset))

        # computes the most represented class

        predictions = np.array(predictions)
        most_represented_class = np.apply_along_axis(_get_most_represented_class, axis= 0, arr= predictions)
        return most_represented_class

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.

        :return:Accuracy of the model.
        """
        y_pred = self.predict(dataset)
        score= accurancy(dataset.y, y_pred)

        return score


