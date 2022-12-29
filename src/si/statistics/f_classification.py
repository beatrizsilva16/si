from scipy.stats import stats
from typing import Tuple, Union
import numpy as np

import sys
sys.path.append("../data")
from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],
                                                Tuple[float, float]]:
    """
    Calculates the score for each feature for classification tasks.
    :param dataset: Dataset object.
    :return: F value for each feature.
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]  # group the dataset by class
    F, p = stats.f_oneway(*groups)

    return F, p
