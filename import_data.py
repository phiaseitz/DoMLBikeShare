import csv
import numpy as np
from sklearn.externals import joblib

# date = []
# time = []
# season = []
# holiday = []
# workingday = []
# weather = []
# temp = []
# atemp = []
# humidity = []
# windspeed = []
# casual = []
# registered = []
# total = []

def read_csv(file_path, has_header = True):
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            line = line.strip().split(",")

            date = line[0][0:line[0].index(' ')]
            date = date.strip().split("-")
            year = date[0]
            month = date[1]
            day = date[2]

            #Start with the index after the space
            time = line[0][line[0].index(' ') + 1: len(line[0])]

            time = time[0:2]

            line.pop(0)
            line.insert(0, time)
            line.insert(0, day)
            line.insert(0, month)
            line.insert(0, year)

            data.append([float(x) for x in line])
            npdata = np.array(data)

    return npdata  

def read_pickle(file_path):
    data = joblib.load(file_path)
    return data

def main ():
    data = read_csv('train.csv')
    print (data)

if __name__ == '__main__':
    main()