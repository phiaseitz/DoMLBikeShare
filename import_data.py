import csv
import numpy as np

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
            date = date.translate(None, '-')
            time = line[0][line[0].index(' '): len(line[0])]
            time = time[0:2]

            line.pop(0)
            line.insert(0, date)
            line.insert(1, time)
            data.append([float(x) for x in line])
            npdata = np.array(data)
    return npdata  

def main ():
    data = read_csv('train.csv')
    print (data)

if __name__ == '__main__':
    main()