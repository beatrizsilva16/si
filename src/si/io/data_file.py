import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ",", features: bool=False, label: bool=False):
    if label:
        data = np.genfromtxt(filename, delimiter=sep)
        x = data[:, :-1]
        y = data[:, -1]

    else:
        x = numpy.genfromtxt(filename, delimiter=sep)
        y = None

    return Dataset(x, y)

def write_data_file(dataset: Dataset, filename: str, label: bool = False, sep: str = ","):
