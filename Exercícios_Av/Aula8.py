main = "../src/si"
from si.io_folder.csv_file import read_csv
from si.linear_model.logistic_regression import LogisticRegression
from si.model_selection.split import train_test_split
from si.feature_extraction.k_mer import KMer
from sklearn.preprocessing import StandardScaler

# Aula 8 - não foi possível realizar os exercícios em Jupyter Notebook

# - Exercise 1: Test the KMER object using the tfbs.csv dataset

# (1) Read csv file - tfbs.csv
tfbs_file = "../datasets/tfbs.csv"
tfbs = read_csv(filename=tfbs_file, sep=",", features=True, label=True)

# (2) Use KMer to get the frequency of each substring in each sequence of the dataset
# Substring size (k): 3
kmer = KMer(k = 3, alphabet='dna')
kmer_dataset = kmer.fit_transform(tfbs)
kmer_dataset.print_dataframe()

# (3) Standardize features by removing the mean and scaling to unit variance.
kmer_dataset.X = StandardScaler().fit_transform(kmer_dataset.X)

# (4) Split dataset in training and testing sets
kmer_train, kmer_test = train_test_split(kmer_dataset)

# (5) Train Logistic Regression model for nucleotide composition
lr = LogisticRegression(l2_penalty= 1, alpha= 0.001, max_iter=2000)
lr.fit(kmer_train)
score = lr.score(kmer_test)

# (6) What is the score obtained?
print(f"The score for nucleotide composition: {score}")

# - Exercise 9: Adapt KMer to calculate the peptide composition

# (1) Read csv file - transporters.csv
transporters_file = "../datasets/transporters.csv"
transporters = read_csv(filename=transporters_file, sep=",", features=True, label=True)

# (2) Use KMer to get the frequency of each substring in each sequence of the dataset
# Substring size (k): 2
kmer_pep = KMer(k = 2, alphabet='peptide')
kmer_pep_dataset = kmer_pep.fit_transform(transporters)

# (3) Standardize features by removing the mean and scaling to unit variance.
kmer_pep_dataset.X = StandardScaler().fit_transform(kmer_pep_dataset.X)

# (4) Split dataset in training and testing sets
kmer_pep_train, kmer_pep_test = train_test_split(kmer_pep_dataset)

# (5) Train Logistic Regression model for peptide composition
lr = LogisticRegression(l2_penalty= 1, alpha= 0.001, max_iter=2000)
lr.fit(kmer_pep_train)
score = lr.score(kmer_pep_test)

# (6) What is the score obtained?
print(f"The score for peptide composition: {score}")

