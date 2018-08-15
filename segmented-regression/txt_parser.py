"""
This file parses the text files from the MASON model.
It parses 'observedInflow', 'storage', 'outflow', 'totalWaterSupply', 'elevation', 'population', and 'shiftFactor'
from MASON txt files, adds 'dates' (mo 1-12), and puts everything into csv files.
"""

import pandas as pd
import os
import time
import datetime


def parse_txt_to_csv(location, shift_list, number_runs):
    dates = pd.DataFrame({'dates': list(range(1,13))*50})
    txt_dict = { 'reservoir-shift_{:.1f}-ts-{:d}'.format(shift, run_num): dates 
                  for shift in shift_list for run_num in range(0, number_runs) }
    colnames = ['observedInflow', 'storage', 'outflow', 'totalWaterSupply', 'waterSupply',
                'elevation', 'population', 'shiftFactor', 'inflow', 'deficit']
    for txt_file in txt_dict.keys():
        print(txt_file)
        txt_dict[txt_file] = pd.concat([dates, pd.read_table(location+txt_file+'.txt', sep=' ',
                                         index_col=False, usecols=colnames)],  axis=1)
        new_loc = "C:\\Users\\maska\\research\\"
        txt_dict[txt_file].to_csv(new_loc+txt_file+'.csv', index=True, index_label='index')


def main():
    start = time.time()

    cwd = 'C:\\Users\\maska\\OneDrive\\Documents\\MASON'
    folder = '20-jun-2018'
    location = cwd + '\\' + folder + '\\'
    shift_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    number_runs = 30

#    shift_list = [0.1]
#    number_runs = 1

    parse_txt_to_csv(location, shift_list, number_runs)

    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()