import numpy as np


class VotingClassifier:
    def __int__(self, fit, predict, score):
        self.fit = fit
        self.predict = predict
        self.score = score

    pass

    self.models = []
    models = [logisticRegression()
              ]


    def Classifier.fit():
        for models in self.models:
            model.fit(dataset)

    def Classifier.predict():

    pass

#Ensemble (models
#self.models = []
#models = [logisticRegression()
#           KNNclassifier (k=fo)

# def fit (dataset): #POO
#   for model in selfmodels:
#       model.fit(dataset)

#def Classifier.predict (models):


def predict (self, dataset: Dataset) -> np.ndarray

    def _get_majority_vote(pred: np.ndarray) -> int:
        labels, counts = np.unique (pred, return_counts=True)
        return labels[np.argmax(counts)] # conta qual tem a contagem maior

    predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
    return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)


def score(self, dataset: Dataset)-> float

pass

#abrir jupyter notebook

brest_bin_dataset = read.csv (...)
from sklearn.processing import StandarScaler
brest_bin_dataset.X = StandarScaler().fit:transform ()

#slipt_dataset

#voting.classifier

knn= KNNclassifier (k=3)
lg =LogisticRegression (l2_penalty=1, alpha=0. max_iter=1000)

