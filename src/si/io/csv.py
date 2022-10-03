import pandas as pd
import numpy as np
from si.data.dataset import Dataset

def read_csv(filename, sep=',', features=False, label=False):
    """
    Parametros: filename, nome do ficheiro ou caminho;
                sep,separador entre valores (default: ',');
                features, booleano indicador se existe ou não uma fila de características;
                label, booleano indicador se há ou não uma fila de etiquetas.

    Returns: pandas dataframe
    """

