import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ",", features: bool=False, label: bool=False) -> Dataset:

    """
    Parametros: filename, nome do ficheiro ou caminho;
                sep,separador entre valores (default: ',');
                features, booleano indicador se existe ou não uma fila de características;
                label, booleano indicador se há ou não uma fila de etiquetas.

    Returns: pandas dataframe
    """

    data = pd.read_csv(filename, delimiter=sep)
    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy() # começar da primeira
        y = data.iloc[:, -1].to_numpy()

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1]
        features = None
        label = None

    else: # sem features nem labels
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(X, y, features, label)

def write_csv(dataset: Dataset, filename: str, sep: str = ',', features: bool = False, label: bool = False) -> None:

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index= False)


