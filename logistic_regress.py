#DoML BikeShare Project
#Shrinidhi Thirumalai, Sophia Sietz, Anne LoVerso

#Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, cross_validation
import import_data

#link to what we want to do!
#http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

np.set_printoptions(threshold=np.inf)

#Functions
def convert_times(data):
	#year, month, day
	[year, month, day] = data[0:3]
	days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
	time = (year - 2011)*365 + month*days_in_month[int(month)] + day
	return [time, data[3:len(data)]]

def load_data(filename):
	#Import the csv
	importedData = import_data.read_pickle(filename)
	print (importedData)

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

def do_learning(XTrain, yTrain,XTest, yTest, variables):
	ridge = linear_model.Ridge(alpha = 0.5)
	ridge.fit(XTrain,yTrain)
	#Prints
	print (ridge.coef_)
	print (ridge.score(XTest, yTest))
	#Return Ridge
	return ridge

def visualize_learn(ridge, XTest, yTest, variables):
	plt.scatter(XTest[:,1], yTest, color = 'black')
	plt.scatter(XTest[:,1], ridge.predict(XTest), color = 'blue')
	plt.show()

def visualize_by_index(XTrain,yTrain,XTest,yTest, indexToPlot):

	indexValsTrain = XTrain[:,indexToPlot]
	indexValsTest	= XTest[:,indexToPlot]

	uniqueValsTrain = np.unique(indexValsTrain)
	uniqueValsTest = np.unique(indexValsTest)

	sumRentalsTrain = [np.sum(yTrain[XTrain[:,indexToPlot]== uniqueVal]) for uniqueVal in uniqueValsTrain]
	sumRentalsTest = [np.sum(yTest[XTest[:,indexToPlot]== uniqueVal]) for uniqueVal in uniqueValsTest]

	plt.figure()
	plt.scatter(uniqueValsTrain,sumRentalsTrain, color = 'blue')
	plt.scatter(uniqueValsTest,sumRentalsTest, color = 'black')
	#plt.show()



def main ():
	#Reading Data
	print ('Reading data')
	data = load_data('training_set.pkl')
	X = data[0]
	y = data[1]

	#Splitting Data
	print ('Splitting data')
	splitData = split_data(X,y,6,0.5)
	wX_train, wX_test, wy_train, wy_test = splitData[0:4]
	hX_train, hX_test, hy_train, hy_test = splitData[4:8]

	#Variables:
	#variables = [1:len(wX_train)] #Everything except year

	#Learning
	#model = do_learning(wX_train,wy_train,wX_test, wy_test)

	#Visualization
	#visualize_learn(model, wX_test, wy_test)
	#visualize_data(wX_train[:,1],wy_train)

	# visualize_by_index(wX_train,wy_train,wX_test,wy_test,3)
	# visualize_by_index(hX_train,hy_train,hX_test,hy_test,3)

	# plt.show()
if __name__ == '__main__':
    main()