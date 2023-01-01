import sys
PATHS = ["si/src/si/data", "src/si/io_folder"]
sys.path.append(PATHS)
from si.feature_extraction.k_mer import KMer
from si.model_selection.split import train_test_split
from si.linear_model.logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from si.io_folder.csv_file import read_csv
tfbs = read_csv('C:/Users/beatr/Mestrado/2ºano/si/datasets/tfbs.csv', sep=',', features=True, label=True)
from si.feature_extraction.k_mer import KMer

kmer = KMer(k=2, alphabet="peptide")
kmer_dataset = kmer.fit_transform(tfbs)

from sklearn.preprocessing import StandardScaler

kmer_dataset.X = StandardScaler().fit_transform(kmer_dataset.X)

#divide o dataset em treino e teste
train_dataset, test_dataset = train_test_split(kmer_dataset)

#treina o modelo LogisticRegression no dataset de composição peptídica
lg = LogisticRegression()
lg.fit(train_dataset)

#qual o score obtido?
lg.score(test_dataset)

