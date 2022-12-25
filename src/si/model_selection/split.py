from typing import Tuple
import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int =42) -> Tuple[Dataset, Dataset]:

    """
    Método para dividir um dataset em dataset de treino e teste
    :param dataset: dataset para dividir em treino e teste
    :param test_size: tamanho do dataset de teste
    :param random_state: seed para gerar permutações
    Return:
    """

    # Set the random state
    np.random.seed(random_state) # Para não obter coisas aleatórias e poder reproduzir esta pipeline sempre que necessário com mesmo resultado

    # Test set size
    n_samples = dataset.shape()[0]

    # Get number of samples in the test set
    n_test = int(n_samples * test_size)

    # Get the dataset permutations
    permutations = np.random.permutations(n_samples)

    # Get the samples in the test set
    test_idxs = permutations[:n_test]

    # Get samples in the training set
    train_idxs = permutations[n_test:]

    # Get the training and testing datasets
    train = Dataset(dataset.x[train_idxs], dataset.y[train_idxs], features=dataset.features, label = dataset.label)

    test = Dataset(dataset.x[test_idxs], dataset.y[test_idxs], features= dataset.features, label=dataset.label)
    return train, test












