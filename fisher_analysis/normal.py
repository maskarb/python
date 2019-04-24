import csv
import random
from statistics import mean, stdev
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from fi import fisher, size_of_state

def to_csv(data: List[List], filename: str) -> None:
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(len(data[0])):
            row = []
            for j in data:
                row.append(j[i])
            w.writerow(row)

def from_csv(filename: str) -> List[List]:
    with open(filename, 'r', newline='\n') as f:
        r = csv.reader(f)
        data = [[], [], []]
        for i in r:
            x, y, e = i
            data[0].append(float(x)); data[1].append(float(y)); data[2].append(float(e))
    return data

dist = random.gauss
n = int(100000)
r1, r2, r = 10, 101, 10
sl = 61

#time = [[i] for i in range(n)]
#s_x = []
#fi_the = []
#
#dis_dic, fi_dic = {}, {}
#for i in range(1, sl):
#    s = i / 10
#    temp = []
#    for _ in range(n):
#        temp.append([dist(0, s)])
#    dis_dic[s] = temp
#    for L in range(r1, r2, r):
#        sos = size_of_state(temp, L)
#        fi = fisher(temp, time, L, 50, sos, 'temp')
#        new_temp = [row[-2] for row in fi]
#        fi_dic[(s, L)] = (mean(new_temp), stdev(new_temp))
#        print((i, L))
#    s_x.append(s)
#    fi_the.append(1/(s**2))
#
#new_dic = {}
#for L in range(r1, r2, r):
#    tempy, tempe = [], []
#    for i in range(1, sl):
#        s = i / 10
#        tempy.append(fi_dic[(s, L)][0])
#        tempe.append(fi_dic[(s, L)][1])
#    data = [s_x, tempy, tempe]
#    new_dic[L] = data
#    to_csv(data, f"{L}-{s}.csv")
    #plt.errorbar(s_x, tempy, yerr=tempe, barsabove=True)

#plt.plot(s_x, fi_the)
#plt.show()


for L in range(r1, r2, r):
    data = from_csv(f"{L}-6.0.csv")
    plt.errorbar(*data, barsabove=True)

y = []
for i in range(1, 61):
    s = i / 10
    y.append(1/s**2)

plt.plot(data[0], y)
plt.yscale("log")
plt.show()