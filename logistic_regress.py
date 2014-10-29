import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, cross_validation
import import_data


def convert_times(data):
	#year, month, day
	[year, month, day] = data[0:3]
	days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
	time = (year - 2011)*365 + month*days_in_month[int(month)] + day
	return [time, data[3:len(data)]]

#http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#link to what we want to do!

def read_data(filename):
	#Import the csv
	importedData = import_data.read_csv(filename)

	#Split into inputs and outputs
	X = importedData[:, 0: importedData.shape[1] - 3]
	y = importedData[:, importedData.shape[1] - 1]
	return (X,y)

def split_data(X,y,modelSplitIndex,testSize):
	#6 is the index of whether or not it is a work day
	#split into working day and non-workind day data sets so that we can learn two different models
	wX = X[X[:,modelSplitIndex] == 1.]
	hX = X[X[:,modelSplitIndex] != 1.]

	wy = y[X[:,modelSplitIndex] == 1.]
	hy = y[X[:,modelSplitIndex] != 1.]

	wX_train , wX_test, wy_train, wy_test = cross_validation.train_test_split(wX, wy, test_size=testSize)
	hX_train , hX_test, hy_train, hy_test = cross_validation.train_test_split(hX, hy, test_size=testSize)
	
	return (wX_train , wX_test, wy_train, wy_test, hX_train , hX_test, hy_train, hy_test)

def visualize_data(xToPlot,yToPlot):
	plt.scatter(xToPlot,yToPlot)
	plt.show()

def do_learning(XTrain, yTrain,XTest, yTest):
	ridge = linear_model.Ridge(alpha = 0.5)
	ridge.fit(XTrain,yTrain)

	print (ridge.coef_)

	print (ridge.score(XTest, yTest))

def visualize_learn(ridge, XTest, yTest):
	plt.scatter(XTest, yTest, color = 'black')
	plt.scatter(ridge.predict(XTest), color = 'blue')

def main ():
	data = read_data('train.csv')
	X = data[0]
	y = data[1]

	splitData = split_data(X,y,6,0.5)

	wX_train = splitData[0]
	wX_test = splitData[1]
	wy_train = splitData[2]
	wy_test = splitData[3]
	hX_train = splitData[4]
	hX_test = splitData[5]
	hy_train = splitData[6]
	hy_test = splitData[7]

	do_learning(wX_train,wy_train,wX_test, wy_test)
	#visualize_data(wX_train[:,1],wy_train)

if __name__ == '__main__':
    main()