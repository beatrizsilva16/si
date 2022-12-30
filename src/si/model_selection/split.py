from random import random
from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.3, random_state: int = 0) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets.
    :param dataset: The dataset to split.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The proportion of the dataset to include in the test split.
    :returns: Tuple[Dataset, Dataset]: The training and test splits.
    """
    # Set the random seed
    np.random.seed(random_state)

    # test set size
    n_samples = dataset.shape()[0]
    split_div = int(n_samples * test_size)

    # get the dataset permutations
    permutations = np.random.permutation(n_samples)

    # get the test and train sets
    test_idx = permutations[:split_div]
    train_idx = permutations[split_div:]

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idx], dataset.y[train_idx], features_name=dataset.features_names,
                    label_name=dataset.label_name)

    test = Dataset(dataset.X[test_idx], dataset.y[test_idx], features_names=dataset.features_names,
                   label_name=dataset.label_name)

    return train, test










