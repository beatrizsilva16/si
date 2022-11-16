
from typing import Tulple
import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int =42 -> ):
    np.random.seed(random_state)
    nsamples = dataset-shape()[]
    n_test = int(n_sample = test_size)
    permutations = np.random.permutation(n_sample)
    teste_idxs = permutations[:n_test]
    train:idxs = permutations[n_testes:]

    #get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.f)
