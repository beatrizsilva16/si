from typing import Dict, Tuple, Callable, Union
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate

def randomized_search_cv(model, dataset: Dataset, parameter_distribution: Dict[str, Tuple], scoring: Callable = None,
                         cv: int = 5, n_iter: int = 10, test_size: float = 0.2):
    """
    It performs a grid search cross validation on a model.
    :param model: the model to cross validate
    :param dataset: Dataset, dataset to cross validate on
    :param parameter_distribution: Dict[str, Tuple], the parameters to use
    :param scoring: callable, the scoring function to use
    :param cv: int, the cross validation folds
    :param n_iter: int, the number of parameter random combinations
    :param test_size: float, the test size
    :return: List[Dict[str, List[float]]], the scores of the model on the dataset
    """
    # scores dictionary
    scores = {'parameters': [], 'seed': [], 'train': [], 'test': []}

    #  checks if parameters exist in the model
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"The {model} does not have parameter {parameter}.")

    #  sets n_iter parameter combinations
    for i in range(n_iter):

        # set the random seed
        random_state = np.random.randint(0, 1000)

        # store the seed
        scores['seed'].append(random_state)

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in parameter_distribution.items():
            # set the combination of parameter and its values to the model
            parameters[parameter] = np.random.choice(value)

        # set the parameters to the model
        for parameter, value in parameters.items():
            setattr(model, parameter, value)

        # performs cross_validation with the combination
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # stores the parameter combination and the obtained scores
        scores['parameters'].append(parameters)
        scores['train'].append(score['train'])
        scores['test'].append(score['test'])

    return scores
