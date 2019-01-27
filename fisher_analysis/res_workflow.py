import csv
from ast import literal_eval
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from scipy.interpolate import InterpolatedUnivariateSpline

STATS = importr('stats')
def sign(a): return (float(a) > 0) - (float(a) < 0)


def min_max_perc(val_list, perc):
    min_val = min(val_list)
    max_val = max(val_list)
    mid_range = (max_val - min_val) * perc + min_val
    return (min_val, max_val, mid_range)


def size_of_state(data_num, window_size):
    d_f = pd.DataFrame(data_num)
    sos = []
    for j in range(len(d_f.columns)):
        sos_temp = []
        for i in d_f.index:
            A = list(d_f[j][i:i+window_size])
            A_1 = [float(i) for i in A if i != 0]
            if len(A_1) == window_size:
                sos_temp.append(np.std(A_1, ddof=1))
        if not sos_temp:
            sos.append(0)
        else:
            sos.append(min(sos_temp) * 2)
    return sos


def fisher(data_num, Time, w_size, w_incr, sost, f_name):
    FI_final, k_init = [], []
    for i in range(0, len(data_num), w_incr):
        data_win = data_num[i:i+w_size]
        if len(data_win) == w_size:
            _bin = []
            for m in range(w_size):
                bin_temp=[]
                for n in range(w_size):
                    if m==n:
                        bin_temp.append('I')
                    else:
                        bin_temp_1=[]
                        for k in range(len(data_win[n])):
                            if (abs(data_win[m][k] - data_win[n][k])) <= sost[k]:
                                bin_temp_1.append(1)
                            else:
                                bin_temp_1.append(0)
                        bin_temp.append(sum(bin_temp_1))
                _bin.append(bin_temp)
            FI = []
            for tl in range(1, 101):
                tl1 = len(sost) * float(tl) / 100
                bin_1, bin_2 = [], []
                for j, _bin_val in enumerate(_bin):
                    if j not in bin_2:
                        bin_1_temp = [j]
                        for i in range(len(_bin[j])):
                            if _bin_val[i] != 'I' and _bin_val[i] >= tl1 and i not in bin_2:
                                bin_1_temp.append(i)
                        bin_1.append(bin_1_temp)
                        bin_2.extend(bin_1_temp)
                prob = [0]
                for i in bin_1:
                    prob.append(float(len(i))/len(bin_2))
                prob.append(0)
                prob_q = []
                for i in prob:
                    prob_q.append(np.sqrt(i))
                FI_temp = 0
                for i in range(len(prob_q) - 1):
                    FI_temp += (prob_q[i] - prob_q[i+1]) ** 2
                FI_temp = 4 * FI_temp
                FI.append(FI_temp)
            for i in range(len(FI)):
                if FI[i] != 8.0:
                    k_init.append(FI.index(FI[i]))
                    break
            FI_final.append(FI)
    if not k_init:
        k_init.append(0)
    for i, FI_val in enumerate(FI_final): #range(0, len(FI_final)):
        FI_final[i].append(
            float(sum(FI_val[min(k_init):len(FI_val)])) /
            len(FI_val[min(k_init):len(FI_val)])
            )
        FI_final[i].append(Time[(i*w_incr+w_size)-1])
    df_FI = pd.DataFrame(FI_final)
    df_FI.to_csv(f"{f_name}_FI.csv", index=False, header=False)
    print("Fisher Done. CSV")
    return FI_final


def read_csv_headers(filename):
    '''Read a csv file, return headers and data in lists'''
    with open(filename, 'r') as out:
        data_csv_reader = csv.reader(out)
        headers = next(data_csv_reader, None)
        raw_data = [row for row in data_csv_reader]
    return headers, raw_data


def read_csv_no_headers(filename):
    '''Read a csv file without headers, return the data in list'''
    with open(filename, 'r') as out:
        data_csv_reader = csv.reader(out)
        raw_data = [row for row in data_csv_reader]
    return raw_data


def fit_data(x, y, df):
    fit1 = STATS.smooth_spline(x, y, df=df) # pylint: disable=E1101
    y_s = list(fit1[1])
    spline = InterpolatedUnivariateSpline(x, y_s)
    derivative = spline.derivative(n=1)
    return spline, derivative


def find_breaks(x, spline, derivative, perc_range):
    breaks = list()
    for i in range(1, len(derivative)):
        if sign(derivative[i]) < sign(derivative[i-1]) and spline[i-1] > perc_range:
            breaks.append(x[i])
    return breaks


def make_plot(x1, y1, x, y, spline, derivative, breaks, ylimit):
    x_s = np.arange(x[0], x[-1], 0.02)
    y_spline, y_derive = find_spline_deriv(x_s, spline, derivative)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, 8)
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 350)

    ln1 = ax2.plot(x1, y1, 'b', linewidth=0.5, alpha=0.5, label='Storage')
    ln2 = ax1.plot(x, y, '.k', markersize=2, label='Fisher Information')
    ln3 = ax1.plot(x_s, y_spline, 'r', label='Smoothing Spline')
    ln4 = []
    for i in breaks:
        ln4.append(ax1.axvline(x=i, color='k', linestyle='-.', linewidth=0.5, label='Breakpoints'))
    # ln5 = ax1.plot(x_s, y_derive, 'g', label='Derivative')
    ln6 = ax1.axhline(y=ylimit, linestyle=':', linewidth=0.35, label='Threshold')

    lns = ln1 + ln2 + ln3 + [ln4[0]]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    ax1.set_xlabel("Time (Months)")
    ax1.set_ylabel("Fisher Information (FI)")
    ax2.set_ylabel("Storage Volume (1000 acre-ft)")

def make_plot_w_dates(datetimes, y1, x, y, spline, derivative, breaks, ylimit):
    dates = matplotlib.dates.date2num(datetimes)

    # find spline/derivative ys, convert x to date
    date_x = convert_x_to_date(x, dates)
    x_s = np.arange(x[0], x[-1], 0.02)
    spline_y = spline(x_s)
    deriva_y = derivative(x_s)
    date_x_s = convert_x_to_date(x_s, dates)
    date_breaks = convert_x_to_date(breaks, dates)

    _, ax1 = plt.subplots()

    ax1.plot_date(date_x, y, '.k')          # plot Fisher Information
    ax1.plot_date(date_x_s, spline_y, 'r')  # plot smoothing spline
    ax1.plot_date(date_x_s, deriva_y, 'g')  # plot derivative

    ax2 = ax1.twinx()
    ax2.plot_date(dates, y1, ':b', linewidth=0.5)   # plot FI variable

    for i in date_breaks:                   # add the breakpoints to plot
        ax1.axvline(x=i, linestyle='--')
    ax1.axhline(y=ylimit, linestyle=':', linewidth=0.35)
    ax1.legend(['Fisher Information', 'Smoothing Spline', 'Derivative', 'Breakpoints'])
    ax1.set_xlabel("Time (Year)")
    ax1.set_ylabel("Fisher Information (FI)")
    ax2.set_ylabel("Percent Storage Volume (%)")

def find_spline_deriv(x, spline, derivative):
    y_spline = spline(x)
    y_derive = derivative(x)
    return y_spline, y_derive

def convert_x_to_date(x, dates):
    start, end, length = dates[0], dates[-1], len(dates)
    date_diff = (end - start) / length
    return np.array(x) * date_diff + start


def FI_smooth(df_FI, df, w_size, w_incr):
    time_index, FI = [w_size], []
    for i, row in enumerate(df_FI):
        FI.append(row[-2])
        time_index.append(time_index[i] + w_incr)
    spline, derivative = fit_data(time_index[:-1], FI, df)
    return spline, derivative, FI, time_index[:-1]



def get_variable_index(vars: list, headers: list):
    return [headers.index(var) for var in vars]



def main():
    file_no_ext = 'fisher_analysis/res-s-0.4-0'
    filename = file_no_ext.split('-')
    f_name = file_no_ext + '.csv'

    w_size = 48
    w_incr = 4
    df = 7
    data_num, Time = [], []

    headers, Data = read_csv_headers(f_name)
    var_list = ['storage', ]
    index_list = get_variable_index(var_list, headers)

    for i, row in enumerate(Data):
        data_num.append([literal_eval(row[i])/1000 for i in index_list])
        Time.append(i)

#    time_list = ['index']
#    time_index = get_variable_index(time_list, headers)
#    for row in Data:
#        Time.append(literal_eval(row[time_index[0]]))

    sost = size_of_state(data_num, w_size)
    df_FI = fisher(data_num, Time, w_size, w_incr, sost, file_no_ext)
    spline, derivative, FI, time_index = FI_smooth(df_FI, df, w_size, w_incr)


    __, __, perc_range = min_max_perc(FI, 0.35)
    xs = np.arange(time_index[0], time_index[-1], 0.02)
    breaks = find_breaks(list(xs), list(spline(xs)), list(derivative(xs)), perc_range)
    make_plot(Time, data_num, time_index, FI, spline, derivative, breaks, perc_range)

    plt.title(f'Falls Lake Reservoir Model\nShift: {filename[2]}, Run: {filename[3]}')
    _file = f'{file_no_ext}_overlay.png'
    plt.savefig(_file, dpi=200)
    plt.close('all') # remove plot from memory



if __name__ == '__main__':
    main()
