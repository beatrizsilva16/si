from typing import Callable
from typing import Dict
from typing import List
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split


def cross_validate(model, dataset: Dataset, scoring: Callable = None, cv: int = 3,
                   test_size: float = 0.2) -> Dict[str, List[float]]:
    """
    It performs cross validation on the given model and dataset.
    It returns the scores of the model on the dataset in a dictionary containing 3 keys :
        1. seeds: The seeds used in the train-test split
        2. train: The scores attained with training data
        3. test: The scores obtained with testing data
    :param model: the model to cross validate
    :param dataset: Dataset, dataset to cross validate on.
    :param scoring: callable, the scoring function to use
    :param cv: int, the cross validation folds
    :param test_size: float, the test size
    :return: Dict[str, List[float]], the scores of the model on the dataset
    """
    # scores dictionary
    scores = {'seeds': [], 'train': [], 'test': []}

    # for each fold
    for i in range(cv):
        # get random seed
        random_state = np.random.randint(0, 1000)

        # store seed
        scores['seeds'].append(random_state)

        # split the dataset
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=random_state)

        # fit the model on the train set
        model.fit(train)

        # score the model on the test set
        if scoring is None:

            # store the train score
            scores['train'].append(model.score(train))

            # store the test score
            scores['test'].append(model.score(test))

        else:
            y_train = train.y
            y_test = test.y

            # store the train score
            scores['train'].append(scoring(y_train, model.predict(train)))

            # store the test score
            scores['test'].append(scoring(y_test, model.predict(test)))

    return scores
