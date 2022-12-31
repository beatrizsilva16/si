from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets.
    :param dataset: The dataset to split.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The proportion of the dataset to include in the test split.
    :returns: Tuple[Dataset, Dataset]: The training and test splits.
    """
    # set random state
    np.random.seed(random_state)

    # get dataset size
    n_samples = dataset.shape()[0]

    # get number of samples in the test set
    n_test = int(n_samples * test_size)

    # get the dataset permutations
    permutations = np.random.permutation(n_samples)

    # get samples in the test set
    test_idxs = permutations[:n_test]  # data until the number of samples defined is reached

    # get samples in the train set
    train_idxs = permutations[n_test:]  # remaining data goes to the train set

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)

    return train, test












