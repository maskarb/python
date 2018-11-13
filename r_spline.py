import csv
import rpy2

import numpy as np
import matplotlib.pyplot as plt
import rpy2.robjects as robjects

from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from scipy.interpolate import InterpolatedUnivariateSpline

stats = importr('stats')
sign = lambda a: (float(a)>0) - (float(a)<0)

def min_max_perc(val_list, perc):
    min_val = min(val_list)
    max_val = max(val_list)
    mid_range = (max_val - min_val) * perc + min_val
    return (min_val, max_val, mid_range)

def read_csv(filename):
    FI, time = [], []
    with open(filename, 'r') as out:
        data_csv_reader = csv.reader(out)
        next(data_csv_reader, None)  # skip the headers
        Data = [row for row in data_csv_reader]
    for row in Data:
        FI.append(eval(row[-2]))
        time.append(eval(row[0])) # uses row index for now
    return time, FI

def fit_data(time, FI, df):
    fit1 = stats.smooth_spline(time, FI, df=df)
    ys = list(fit1[1])
    spline = InterpolatedUnivariateSpline(time, ys)
    derivative = spline.derivative(n=1)
    return spline, derivative

def find_breaks(x, y_deriv, fit, perc_FI):
    breaks = list()
    for i in range(1, len(y_deriv)):
        if sign(y_deriv[i]) < sign(y_deriv[i-1]) and fit[i-1] > perc_FI:
            breaks.append(x[i])
    return breaks

def make_plot(x, y, spline, derivative, breaks, ylimit):
    plt.plot(x, y, '.k')
    xs = np.arange(0, len(y), 0.02)
    plt.plot(xs, spline(xs), 'b')
    plt.plot(xs, derivative(xs), 'g')
    for i in breaks:
        plt.axvline(x=i, linestyle='--')
    plt.axhline(y=ylimit, linestyle=':', linewidth=0.35)
    plt.legend(['Fisher Information', 'Smoothing Spline', 'Derivative', 'Breakpoints'])


def main():

    # filename = 'dados_jun2018.csv_win_48_incr_3_smooth_6_FI.csv'
    # filename = 'reservoir-shift_0.1-ts-0.csv_win_48_incr_4_smooth_6_FI.csv'
    filename = 'rs1_ts1_bp.csv_win_48_incr_4_smooth_6_FI.csv'
    time, FI = read_csv(filename)
    __, __, perc_FI = min_max_perc(FI, 0.25)

    time_arr = FloatVector(time)
    fish_arr = FloatVector(FI)

    xs = np.arange(0, len(time), 0.02)
    for i in range(5, 16):
    # for i in [25.85]:
        df = i
        spline, derivative = fit_data(time_arr, fish_arr, df)
        breaks = find_breaks(list(xs), list(derivative(xs)), list(spline(xs)), perc_FI)
        make_plot(time, FI, spline, derivative, breaks, perc_FI)

        plt.title('DF = %.2f' % df)
        # _file = 'dados_df_%s.png' % df
        _file = 'rs1_ts1_bp_df_%s.png' % df
        plt.savefig(_file)
        plt.gcf().clear() # clear old plot
        print(i)


if __name__ == '__main__':
    main()