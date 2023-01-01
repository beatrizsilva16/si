# Não foi possível fazer no notebook.

from si.io_folder.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.neighbors.knn_classifier import KNNClassifier
from si.linear_model.logistic_regression import LogisticRegression
from si.ensemble.voting_classifier import VotingClassifier
from si.ensemble.stacking_classifier import StackingClassifier
from si.statistics.euclidean_distance import euclidean_distance
from sklearn.preprocessing import StandardScaler

# - Exercise: Test the object VotingClassifier using the breast-bin.csv dataset

# (1) Read csv file - breast-bin.csv
breast_file = "../datasets/breast-bin.csv"
breast = read_csv(filename=breast_file, sep=",", features=True, label=True)

# (2) Standardize features by removing the mean and scaling to unit variance.
breast.X = StandardScaler().fit_transform(breast.X)

# (3) Split dataset in training and testing sets
breast_train, breast_test = train_test_split(breast)

# (4) Model KNNClassifier
knn = KNNClassifier(k=3, distance=euclidean_distance)

# (5) Logistic Regression
lr = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=2000)

# (6) Create VotingClassifier ensemble model using the previous classifiers
voting = VotingClassifier([knn, lr])

# (7) Train the model
voting.fit(breast_train)

# (8) What is the score obtained?
score = voting.score(breast_test)
print(f"The voting score is {score}")

# - Exercise 6: Test the object StackingClassifier using the breast-bin.csv dataset

# (1) Read csv file - breast-bin.csv

breast_file = "../datasets/breast-bin.csv"
breast = read_csv(filename=breast_file, sep=",", features=True, label=True)

# (2) Standardize features by removing the mean and scaling to unit variance.
breast.X = StandardScaler().fit_transform(breast.X)

# (3) Split dataset in training and testing sets
breast_train, breast_test = train_test_split(breast)

# (4) Model KNNClassifier
knn = KNNClassifier(k=3, distance=euclidean_distance)

# (5) Logistic Regression
lr = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=2000)

# (6) Create final KNNClassifier model
knn_final = KNNClassifier(k=2, distance=euclidean_distance)

# (7) Create VotingClassifier ensemble model using the previous classifiers and using final model knn
stacking = StackingClassifier([lr, knn], knn_final)

# (7) Train the model
stacking.fit(breast_train)

# (8) What is the score obtained?
score = stacking.score(breast_test)
print(f"The stacking score is {score}")
