import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    """
    It implements an ensemble classifier model which uses a stack of models to train a final classifier.
    """
    def __init__(self, models: list, final_model):
        """
        Initialize the ensemble classifier.
        :param models: array-like, shape = [n_models], different models for the ensemble
        :param final_model: classifier (KNNClassifier), the final classifier to be used
        """
        # parameters
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        It fits the models to the dataset.
        :param dataset: Dataset, dataset object to fit the models to.
        :return: self, fitted model
        """
        # first it trains the models
        for model in self.models:
            model.fit(dataset)

        # gets the predictions of the model
        predictions = np.array([model.predict(dataset) for model in self.models])

        # creates dataset with transpose of the predictions
        train_dataset = Dataset(predictions.T, dataset.y)

        # fits the final model
        self.final_model.fit(train_dataset)
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts of the models and returns the final model prediction.
        :param dataset: Dataset, dataset to predict the labels of
        :return: np.ndarray, the final model prediction
        """
        # gets the predictions of the model
        predictions = np.array([model.predict(dataset) for model in self.models])

        # creates dataset
        test_dataset = Dataset(predictions.T, dataset.y)

        # returns the predictions
        return self.final_model.predict(test_dataset)

    def score(self, dataset: Dataset) -> float:
        """
        It calculates score, which means the accuracy of the model.
        :return: float, accuracy of the model.
        """
        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)
        return score
