main = "../src/si"
from si.io_folder.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.neural_networks.layer import Dense, SigmoidActivation, ReLUActivation
from si.neural_networks.nn import NN
from sklearn.preprocessing import StandardScaler

# Aula 12
# Exercise 12.2. Build a neural network model suitable to the dataset breast-bin.csv

# read csv file - breast-bin.csv
breast_file = "../datasets/breast-bin.csv"
breast_ds = read_csv(filename=breast_file, sep=",", features=True, label=True)

# standardize features by removing the mean and scaling to unit variance.
breast_ds.X = StandardScaler().fit_transform(breast_ds.X)

# split dataset in training and testing sets
breast_train, breast_test = train_test_split(breast_ds)

# see the shape of an array.
breast_ds_shape = breast_train.shape()
print(f"Train set shape: {breast_ds_shape}")
# R: Train set shape: (559, 9)

# build a suitable neural network model
layer1_breast = Dense(input_size=9, output_size=9)
layer2_breast = Dense(input_size=9, output_size=5)
layer3_breast = Dense(input_size=5, output_size=1)

layer1_breast_activation = ReLUActivation()
layer2_breast_activation = ReLUActivation()
layer3_breast_activation = SigmoidActivation()

breast_model = NN(layers=[layer1_breast, layer1_breast_activation, layer2_breast, layer2_breast_activation,
                          layer3_breast, layer3_breast_activation])

# fit and predict model
breast_model.fit(dataset=breast_train)
breast_model.predict(dataset=breast_train)

# Exercise 12.3. Build a neural network model suitable to the dataset cpu.csv

# Read csv file - cpu.csv
cpu_file = "../datasets/cpu.csv"
cpu_ds = read_csv(filename=cpu_file, sep=",", features=True, label=True)

# Standardize features by removing the mean and scaling to unit variance.
cpu_ds.X = StandardScaler().fit_transform(cpu_ds.X)

# Split dataset in training and testing sets
cpu_train, cpu_test = train_test_split(cpu_ds, test_size=0.3, random_state=2)

# see the shape of an array.
cpu_ds_shape = cpu_train.shape()
print(f"cpu train set of shape: {cpu_ds_shape}")
# R: cpu train set shape: (147, 6)

# build a suitable neural network model
layer1_cpu = Dense(input_size=6, output_size=6)
layer2_cpu = Dense(input_size=6, output_size=4)
layer3_cpu = Dense(input_size=4, output_size=1)

layer1_cpu_activation = ReLUActivation()
layer2_cpu_activation = ReLUActivation()
layer3_cpu_activation = SigmoidActivation()

cpu_model = NN(layers=[layer1_cpu, layer1_cpu_activation, layer2_cpu, layer2_cpu_activation,
                          layer3_cpu, layer3_cpu_activation])

# fit and predict model
cpu_model.fit(dataset=cpu_train)
cpu_model.predict(dataset=cpu_train)