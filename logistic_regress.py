#DoML BikeShare Project
#Shrinidhi Thirumalai, Sophia Seitz, Anne LoVerso

#Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import import_data

#Useful Link
#http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

#Settings:
CATEG = ['year', 'month', 'day', 'time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
DISCLUDE = []
np.set_printoptions(threshold=np.inf)


#Functions
def load_data(filename):
	#Import the csv
	importedData = import_data.read_pickle(filename)
	#print (importedData)

	#Split into inputs and outputs
	X = importedData[:, 0: importedData.shape[1] - 3]
	y = importedData[:, importedData.shape[1] - 1]
	return (X,y)

def split_data(X,y,modelSplitIndex,testSize):
	#split into working day and non-working day data sets so that we can learn two different models

	#Splitting on workday
	wX = X[X[:,modelSplitIndex] == 1.]
	hX = X[X[:,modelSplitIndex] != 1.]

	wy = y[X[:,modelSplitIndex] == 1.]
	hy = y[X[:,modelSplitIndex] != 1.]

	#Deleting workday
	for data in [wX, hX]:
		data = np.delete(data, modelSplitIndex, 1)
	CATEG.pop(modelSplitIndex)

	#Test and Train Split
	wX_train , wX_test, wy_train, wy_test = cross_validation.train_test_split(wX, wy, test_size=testSize)
	hX_train , hX_test, hy_train, hy_test = cross_validation.train_test_split(hX, hy, test_size=testSize)
	
	return (wX_train , wX_test, wy_train, wy_test, hX_train , hX_test, hy_train, hy_test)

def visualize_data(xToPlot,yToPlot):
	plt.scatter(xToPlot,yToPlot)
	#plt.show()

def do_learning(XTrain, yTrain,XTest, yTest):
	#Fits Data
	ridge = linear_model.Ridge(alpha = 0.05)
	# ridge = make_pipeline(PolynomialFeatures(2), linear_model.Ridge())
	# ridge = SVR(kernel='poly', C=1e3, degree=4)
	ridge.fit(XTrain,yTrain)

	#Prints Some Results
	print 'Coefficients: ', (ridge.coef_)
	print 'Score: ', (ridge.score(XTest, yTest))

	#Return Ridge
	return ridge

def visualize_by_index(XTrain,yTrain,XTest,yTest, indexToPlot):
	title = 'Bike Count by ' + CATEG[indexToPlot]

	indexValsTrain = XTrain[:,indexToPlot]
	indexValsTest	= XTest[:,indexToPlot]

	uniqueValsTrain = np.unique(indexValsTrain)
	uniqueValsTest = np.unique(indexValsTest)

	sumRentalsTrain = [np.sum(yTrain[XTrain[:,indexToPlot]== uniqueVal]) for uniqueVal in uniqueValsTrain]
	sumRentalsTest = [np.sum(yTest[XTest[:,indexToPlot]== uniqueVal]) for uniqueVal in uniqueValsTest]

	plt.figure()
	plt.scatter(uniqueValsTrain,sumRentalsTrain, color = 'blue')
	plt.scatter(uniqueValsTest,sumRentalsTest, color = 'black')
	plt.suptitle(title)
	#plt.show()

def visualize_learn(ridge, XTest, yTest, indexToPlot):
	#Creates Scatter Plot of Predictions
	visualize_by_index(XTest, yTest, XTest, ridge.predict(XTest), indexToPlot)
	# plt.scatter(XTest[:,CATEG.index('time')], yTest, color = 'black')
	# plt.scatter(XTest[:,CATEG.index('time')], ridge.predict(XTest), color = 'blue')
	#plt.show()

def main ():
	#Reading Data
	print ('Reading data')
	data = load_data('training_set.pkl')
	X = data[0]
	y = data[1]

	#Disclude:
	disclude = [DISCLUDE.index(categ) for categ in DISCLUDE]
	X = np.delete(X, disclude, 1)
	for index in disclude:
		CATEG.pop(index)

	#Splitting Data
	splitData = split_data(X,y,CATEG.index('workingday'),0.5)
	wX_train, wX_test, wy_train, wy_test = splitData[0:4]
	hX_train, hX_test, hy_train, hy_test = splitData[4:8]

	#Learning
	wmodel = do_learning(wX_train,wy_train,wX_test, wy_test)
	hmodel = do_learning(hX_train,hy_train,hX_test, hy_test)


	#Visualization
	visualize_learn(wmodel, wX_test, wy_test, CATEG.index('time'))
	visualize_learn(hmodel, hX_test, hy_test, CATEG.index('time'))

	# visualize_data(wX_train[:,1],wy_train)
	# visualize_by_index(wX_train,wy_train,wX_test,wy_test, CATEG.index('temp'))
	# visualize_by_index(hX_train,hy_train,hX_test,hy_test,CATEG.index('temp'))
	plt.show()

if __name__ == '__main__':
    main()