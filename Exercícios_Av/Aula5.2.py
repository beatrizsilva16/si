
main = "../src/si"
from si.io_folder.csv_file import read_csv
from si.linear_model.logistic_regression import LogisticRegression
from si.linear_model.ridge_regression import RidgeRegression
from si.model_selection.split import train_test_split
from sklearn.preprocessing import StandardScaler

# Aula 5 - não foi possível realizar os exercícios em Jupyter Notebook

# Exercise 6
# - Use the dataset cpu.csv and the package matplotlib to visualize the behavior of the cost function
# in the RidgeRegression model.

# Read csv file - cpu.csv
cpu = read_csv('C:/Users/beatr/Mestrado/2ºano/si/datasets/cpu.csv', sep=',', features=True, label=True)

# Standardize features by removing the mean and scaling to unit variance.
cpu.X = StandardScaler().fit_transform(cpu.X)

# Split dataset in training and testing sets
cpu_train, cpu_test = train_test_split(cpu, test_size=0.3, random_state=2)

# RidgeRegression
rr = RidgeRegression(max_iter=2000)
rr.fit(cpu_train)
rr.predict(cpu_test)
rr.score(cpu_test)
rr.cost(cpu_test)
rr.cost_function_plot()

# - Use the dataset breast-bin.csv and the package matplotlib to visualize the behavior of the cost function
# in the LogisticRegression model.

# Read csv file - breast-bin.csv
breast_file = "../datasets/breast-bin.csv"
breast = read_csv(filename=breast_file, sep=",", features=True, label=True)

# Standardize features by removing the mean and scaling to unit variance.
breast.X = StandardScaler().fit_transform(breast.X)

# Split dataset in training and testing sets
breast_train, breast_test = train_test_split(breast, test_size=0.3, random_state=2)

# Logistic Regression
lr = LogisticRegression(l2_penalty= 1, alpha= 0.001, max_iter=2000)
lr.fit(breast_train)
lr.predict(breast_test)
lr.score(breast_test)
lr.cost(breast_test)
lr.cost_function_plot()