import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import import_data

#Import the csv
importedData = import_data.read_csv('train.csv')

#Split into inputs and outputs
X = importedData[:, 0: importedData.shape[1] - 3]
y = importedData[:, importedData.shape[1] - 1]

#wX = np.split(X, np.where(X[:,3] == 1.) [0][0:])
#split into working day and non-workind day data sets so that we can learn two different models
wX = X[X[:,4] == 1.]
hX = X[X[:,4] != 1.]

wy = y[X[:,4] == 1.]
hy = y[X[:,4] != 1.]

#http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#link to what we want to do!



print (wy)


# print (X)
# print (y)