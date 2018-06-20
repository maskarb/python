import pandas as pd
import numpy as np
import os
import datetime

cwd = 'C:\\Users\\maska\\OneDrive\\Documents\\MASON'
folder = '25-May-2018'
location = cwd + '\\' + folder + '\\'

file_dict = { '{:.1f}_shift_ts-{:d}_all_params.csv'.format(shift/10.0, run_num): 0 for shift in range(1, 11) for run_num in range(0, 10) }


for csvfile in file_dict.keys():
    file_dict[csvfile] = pd.read_csv(location + csvfile)

# df = pd.read_csv(location + '1.0_shift_ts-9_all_params.csv')

filename = '1.0_shift_ts-0_all_params.csv'
months = file_dict[filename]['dates']
storage= file_dict[filename]['storage_asis']

print(filename, months, storage)

def drought_stage(month):
    return {
        1 : [40, 30, 25],
        2 : [50, 35, 25],
        3 : [65, 45, 30],
        4 : [85, 60, 35],
        5 : [75, 55, 35],
        6 : [65, 45, 30],
        7 : [55, 45, 25],
        8 : [50, 40, 25],
        9 : [45, 35, 25],
        10: [40, 30, 25],
        11: [35, 30, 25],
        12: [35, 30, 25],
    }[month]

def recission(month):
    return {
        1 : [60,  50, 45],
        2 : [70,  55, 45],
        3 : [85,  65, 50],
        4 : [100, 80, 55],
        5 : [95,  75, 55],
        6 : [85,  65, 50],
        7 : [75,  65, 45],
        8 : [70,  60, 45],
        9 : [65,  55, 45],
        10: [60,  50, 45],
        11: [55,  50, 45],
        12: [55,  50, 45],
    }

stage1 = [False] * len(file_dict[filename])
stage2 = [False] * len(file_dict[filename])
stage3 = [False] * len(file_dict[filename])

for i in range(len(file_dict[filename])):
    if storage[i] <= drought_stage(months[i])[0]:
        stage1[i] = True
    if storage[i] <= drought_stage(months[i])[1]:
        stage2[i] = True
    if storage[i] <= drought_stage(months[i])[2]:
        stage3[i] = True
file_dict[filename]['stage1'] = stage1
file_dict[filename]['stage2'] = stage2
file_dict[filename]['stage3'] = stage3
print(file_dict[filename])


path = os.getcwd()
subfolder = '\\segmented-regression'
today = datetime.datetime.today().strftime('%Y-%m-%d')
path_to_make = path + subfolder + '\\' + today

while True:
    try:
        os.mkdir(path_to_make)
        print('make dir')
    except:
        break


for key in file_dict.keys():
    file_dict[key].to_csv(path_to_make+'\\'+key+'.csv')