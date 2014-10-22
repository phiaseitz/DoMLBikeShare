import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

date = []
time = []
season = []
holiday = []
workingday = []
weather = []
temp = []
atemp = []
humidity = []
windspeed = []
casual = []
registered = []
total = []

with open('train.csv', 'rb') as csvfile:
	bikereader = csv.reader(csvfile, delimiter=',', quotechar='|')
  	for row in bikereader:
  		for column in row:
  			
  		X.append[]