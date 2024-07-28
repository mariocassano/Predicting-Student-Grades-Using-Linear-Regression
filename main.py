import numpy as np
import pandas as pd
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt

# IMPORT DATASET AND RANDOM SHUFFLING
students = pd.read_csv('students.csv')
students = students.sample(frac=1, random_state=1)

x = students.drop(['GPA', 'GradeClass'], axis=1)
y = students['GPA']

# HOLDOUT SPLITTING
train_index = round(len(x) * 0.7)

x_train = x[:train_index]
y_train = y[:train_index]

x_test = x[train_index:]
y_test = y[train_index:]

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

# Z-SCORE NORMALIZATION
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# ADDING BIAS TERM
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]

# TRAINING
model = LinearRegression(alpha=0.01, steps=1500, features=x_train.shape[1])
cost = model.train(x_train, y_train)

# TEST
result = model.predict(x_test)

# EVALUATION
training_error_perc = round(np.mean(np.abs(cost)) * 100, 2)
test_error_perc = round(np.mean(np.abs(result - y_test)) * 100, 2)

# STAT
print('Precisione sul training set: ' + str(round(100 - training_error_perc, 2)) + ' %')
print('Precisione sul test-set: ' + str(round(100 - test_error_perc, 2)) + ' %')

# PLOT COST FUNCTION
epochs = range(1, model.steps + 1)
plt.plot(epochs, cost)
plt.xlabel("Training Epochs")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.savefig("result")
plt.show()
