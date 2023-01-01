import itertools

from typing import Callable, Tuple, Dict, List, Any
from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model, dataset: Dataset, parameter_grid: Dict[str, Tuple], scoring: Callable = None,
                   cv: int = 5, test_size: float = 0.2)-> List[Dict[str, Any]]:
    """
    It performs a grid search cross validation on a model.
    :param model: the model to cross validate
    :param dataset: Dataset, dataset to cross validate on
    :param parameter_grid: Dict[str, Tuple], the parameter grid to use
    :param scoring: callable, the scoring function to use
    :param cv: int, the cross validation folds
    :param test_size: float, the test size
    :return: List[Dict[str, List[float]]], the scores of the model on the dataset
    """
    # validate the parameter grid
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []

    # for each combination
    for combination in itertools.product(*parameter_grid.values()):

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # add the parameter configuration
        score['parameters'] = parameters

        # add the score
        scores.append(score)

    return scores