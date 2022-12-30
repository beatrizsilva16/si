from itertools import permutations
from typing import Tuple
from typing import Callable
from si.data.dataset import Dataset
import numpy as np


def train_test_split(dataset: Dataset, test_size: float, random_state: int =42):

    """
    Método para dividir um dataset em dataset de treino e teste
    :param dataset: dataset para dividir em treino e teste
    :param test_size: tamanho do dataset de teste
    :param random_state: seed para gerar permutações
    Return:
    """

    # Set the random state
    data = dataset.shape()[0]
    np.random.seed(random_state) # Para não obter coisas aleatórias e poder reproduzir esta pipeline sempre que necessário com mesmo resultado

    # Test set size
    teste_size =test_size
    n_samples = dataset.shape()[0]

    # Get number of samples in the test set
    n_test = int(data * test_size)

    # Get the dataset permutations
    permutations = np.random.permutation(data)

    # Get the samples in the test set
    test_idxs = permutations[:n_test]

    # Get samples in the training set
    train_idxs = permutations[n_test:]

    # Get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label = dataset.label)

    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features= dataset.features, label=dataset.label)
    return train, test












