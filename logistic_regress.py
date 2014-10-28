import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, cross_validation
import import_data

#Import the csv
importedData = import_data.read_csv('train.csv')

def convert_times(data):
	#year, month, day
	[year, month, day] = data[0:3]
	days_in_month = [1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31]
	time = (year - 2011)*365 + month*days_in_month[int(month)] + day
	return [time, data[3:len(data)]]

#Split into inputs and outputs
X = importedData[:, 0: importedData.shape[1] - 3]
y = importedData[:, importedData.shape[1] - 1]

#wX = np.split(X, np.where(X[:,3] == 1.) [0][0:])
#split into working day and non-workind day data sets so that we can learn two different models
wX = X[X[:,6] == 1.]
hX = X[X[:,6] != 1.]

wy = y[X[:,6] == 1.]
hy = y[X[:,6] != 1.]

wX_train , wX_test, wy_train, wy_test = cross_validation.train_test_split(wX, wy, test_size=.5)
hX_train , hX_test, hy_train, hy_test = cross_validation.train_test_split(hX, hy, test_size=.5)

#http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#link to what we want to do!

print (wy)


# print (X)
# print (y)

plt.plot(wX[:,0],wy)
plt.show()