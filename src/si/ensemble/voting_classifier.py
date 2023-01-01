import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class VotingClassifier:
    """
    It implements an ensemble classifier model which uses the majority vote to predict the class labels.
    """
    def __init__(self, models):
        """
        Initialize the ensemble classifier.
        :param models: array-like, shape = [n_models], different models for the ensemble.
        """
        #parameter
        self.models = models

    def fit(self, dataset: Dataset):
        """
        It fits the models according to the given training data.
        :param dataset: Dataset, training dataset
        :return: self, fitted model
        """
        for model in self.models:
            model.fit(dataset)
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predict class labels for samples in X.
        :param dataset: Dataset, testing dataset
        :return: array-like, shape = [n_samples], predicted class labels
        """
        #helper function
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            It returns the majority vote of the given predictions
            :param pred: np.ndarray, predictions to get the majority vote of
            :return: int, the majority vote of the given predictions
            """
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)
            # return_counts if True returns the number of times each unique item appears in the array
            return labels[np.argmax(counts)]
            # argmax obtains the one label that has more counts

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the mean accuracy on the given test data and labels.
        :param dataset: Dataset, test data
        :return: float, mean accuracy obtained
        """
        return accuracy(dataset.y, self.predict(dataset))

