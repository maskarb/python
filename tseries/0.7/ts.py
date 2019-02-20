import csv
import statistics as stat
import os

import matplotlib.pyplot as plt

from historical_data import flows

def read_csv_headers(filename):
    '''Read a csv file, return headers and data in lists'''
    with open(filename, 'r') as out:
        data_csv_reader = csv.reader(out)
        headers = next(data_csv_reader, None)
        raw_data = [row for row in data_csv_reader]
    return raw_data

mypath = os.getcwd()
files = [ f for f in os.listdir(mypath) if f.endswith('.csv') ]

data = {k: [] for k in range(600)}
raw = read_csv_headers(files[0])
for i, row in enumerate(raw):
    data[i].append(float(row[3]))

for f in files:
    raw = read_csv_headers(f)
    for i, row in enumerate(raw):
        data[i].append(float(row[0]))

means = {k: 0 for k in range(600)}

count = 0
for key in data.keys():
    means[key] = [
        data[key][0], 
        stat.mean(flows[key % 12]) * data[key][0], 
        stat.mean(data[key][1:])]

x, y1, y2 = list(range(600)), list(range(600)), list(range(600))
for i in range(600):
    y1[i] = means[i][1]
    y2[i] = means[i][2]

plt.plot(x, y1, label="historical means")
plt.plot(x, y2, label="reconstructed")
plt.xlabel("Time")
plt.ylabel("Inflow")
plt.legend()
plt.show()