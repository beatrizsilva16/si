import pandas as pd
import sys
from si.data.dataset import Dataset
sys.path.append("../data")


def read_csv(filename: str,
             sep: str = ",",
             features: bool = False,
             label: bool = False) -> Dataset:
    """
    Parametros: filename, nome do ficheiro ou caminho;
                sep,separador entre valores (default: ',');
                features, booleano indicador se existe ou não uma fila de características;
                label, booleano indicador se há ou não uma fila de etiquetas.

    Returns: pandas dataframe
    """

    data = pd.read_csv(filename, sep=sep)

    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()  # começar da primeira
        y = data.iloc[:, -1].to_numpy()

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None

    else:  # sem features nem labels
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(X, y, features, label=label)


def write_csv(filename: str,
              dataset: Dataset,
              sep: str = ',',
              features: bool = False,
              label: bool = False) -> None:
    """
        Writes a Dataset object to a csv file
        Parameters
        ----------
        filename : str
            Path to the file
        dataset : Dataset
            The dataset object
        sep : str, optional
            The separator used in the file, by default ','
        features : bool, optional
            Whether the file has a header, by default False
        label : bool, optional
            Whether the file has a label, by default False
    """

    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False)
