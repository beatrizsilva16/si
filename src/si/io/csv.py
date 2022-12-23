import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ",", features: bool=False, label: bool=False):

    """
    Parametros: filename, nome do ficheiro ou caminho;
                sep,separador entre valores (default: ',');
                features, booleano indicador se existe ou não uma fila de características;
                label, booleano indicador se há ou não uma fila de etiquetas.

    Returns: pandas dataframe
    """

    dataframe = pd.read_csv(filename, delimiter=sep)
    if features and label:
        features = dataframe.columns[:1]
        x = dataframe.iloc[1:, 0:, -1].to_numpy()
        y = dataframe.iloc[:, -1].to_numpy()
        features = None
        label = None

    elif features and not label:
        features = dataframe.columns
        y = None
    elif not features and label:
        label = dataframe.columns[-1]
        y = dataframe.iloc[:, -1]
        data = dataframe.iloc[:, :-1]
    else:
        y = None
    return Dataset(data, y, features, label)

def write_csv(dataset: Dataset, filename: str, sep: str = ',', features: bool = False, label: bool = None):

    if features and label:
        dataframe = pd.DataFrame(dataset.x, columns=dataset.features)
        dataframe[dataset.labels] = dataset.y
        dataframe.to_csv(filename, sep=sep, index=False)

    elif features:
        dataframe = pd.DataFrame(dataset.x, columns=dataset.features)
        dataframe.to_csv(filename, sep=sep, index=False)

    elif label:
        dataframe = pd.DataFrame(dataset.y, columns=dataset.labels)
        dataframe.to_csv(filename, sep=sep, index=False)

    else:
        dataframe = pd.DataFrame(dataset.x)
        dataframe.to_csv(filename, sep=sep, index=False)
