#DoML BikeShare Project
#Shrinidhi Thirumalai, Sophia Sietz, Anne LoVerso

#Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, cross_validation
import import_data

#link to what we want to do!
#http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

Categories = {'year': 0, 'month': 1, 'day': 2, 'time': 3, 'season': 4, 'holiday': 5, 'workingday': 6, 'weather': 7, 'temp': 8, 'atemp': 9, 'humidity': 10, 'windspeed': 11, 'casual': 12, 'registered': 13, 'count': 14}

#Functions
def convert_times(data):
	#year, month, day
	[year, month, day] = data[0:3]
	days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
	time = (year - 2011)*365 + month*days_in_month[int(month)] + day
	return [time, data[3:len(data)]]

def read_data(filename):
	#Import the csv
	importedData = import_data.read_csv(filename)

	#Split into inputs and outputs
	X = importedData[:, 0: importedData.shape[1] - 3]
	y = importedData[:, importedData.shape[1] - 1]
	return (X,y)

def split_data(X,y,modelSplitIndex,testSize):
	#split into working day and non-workind day data sets so that we can learn two different models

	#Splitting on workday
	wX = X[X[:,modelSplitIndex] == 1.]
	hX = X[X[:,modelSplitIndex] != 1.]

	wy = y[X[:,modelSplitIndex] == 1.]
	hy = y[X[:,modelSplitIndex] != 1.]

	#Deleting workday
	data = [np.delete(data, modelSplitIndex, 1)for data in [wX, hX, wy, hy]]

	#Test and Train Split
	wX_train , wX_test, wy_train, wy_test = cross_validation.train_test_split(wX, wy, test_size=testSize)
	hX_train , hX_test, hy_train, hy_test = cross_validation.train_test_split(hX, hy, test_size=testSize)
	
	return (wX_train , wX_test, wy_train, wy_test, hX_train , hX_test, hy_train, hy_test)

def visualize_data(xToPlot,yToPlot):
	plt.scatter(xToPlot,yToPlot)
	plt.show()

def do_learning(XTrain, yTrain,XTest, yTest, disclude):
	ridge = linear_model.Ridge(alpha = 0.5)
	for data in [XTrain, yTrain, XTest, yTest]:
		data = np.delete(data, disclude, 1)
	ridge.fit(XTrain,yTrain)
	#Prints
	print (ridge.coef_)
	print (ridge.score(XTest, yTest))
	#Return Ridge
	return ridge

def visualize_learn(ridge, XTest, yTest, disclude):
	for data in [XTest, yTest]:
		data = np.delete(data, disclude, 1)
	plt.scatter(XTest[:,1], yTest, color = 'black')
	plt.scatter(XTest[:,1], ridge.predict(XTest), color = 'blue')
	plt.show()

def main ():
	#Reading Data
	data = read_data('train.csv')
	X = data[0]
	y = data[1]

	#Splitting Data
	splitData = split_data(X,y,Categories['workingday'],0.5)
	wX_train, wX_test, wy_train, wy_test = splitData[0:4]
	hX_train, hX_test, hy_train, hy_test = splitData[4:8]

	#disclude:
	disclude = [0] #Everything except year

	#Learning
	model = do_learning(wX_train,wy_train,wX_test, wy_test, disclude)

	#Visualization
	visualize_learn(model, wX_test, wy_test, disclude)
	#visualize_data(wX_train[:,1],wy_train)

if __name__ == '__main__':
    main()