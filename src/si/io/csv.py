import pandas as pd
import numpy as np
from si.data.dataset import Dataset

def read_csv(filename, str,
             sep str
             sep=',', features=False, label=False):
    """
    Parametros: filename, nome do ficheiro ou caminho;
                sep,separador entre valores (default: ',');
                features, booleano indicador se existe ou não uma fila de características;
                label, booleano indicador se há ou não uma fila de etiquetas.

    Returns: pandas dataframe
    """

    data = pd read_csv(filename, sep= sep)
    if features and label:
        features = data coluna [:1]
        label = data coluna [:1]
        x = data iloc [1, 0:-1]


    data = pd.DataFrame(dataset.x)
    if features:
        data.coluna = dataset.features

    if label:
        data[]
