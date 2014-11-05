from sklearn.externals import joblib
import numpy as np
import import_data

training_set = import_data.read_csv('train.csv')

joblib.dump(training_set, 'training_set.pkl')
