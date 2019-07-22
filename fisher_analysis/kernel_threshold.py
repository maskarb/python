import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.stats import gaussian_kde as kde

from kernel_functions import kernel_concise
# N - number of source points
# X - 1 x N matrix of N source points
# eps - accuracy parameter
# h - optimal bandwidth estimated

INF = float("inf")
historical = pd.read_csv('historical_storage.csv')


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def get_dataframe(filename):
    return pd.read_csv(filename)


def get_min_max(lis):
    return min(lis), max(lis)


def get_kde_probs(kde, datapoints):
    datapoints.append(INF)
    return [kde.integrate_box_1d(-INF, d) for d in datapoints]


def get_uni_kde(data, bw_method="silverman"):
    return kde(data, bw_method=bw_method)

def get_value(x, k, p):
    return p - k.integrate_box_1d(-np.inf, x)

def get_bin_range(num_bins, minim, maxim):
    return np.linspace(minim, maxim, num_bins + 1)


def bin_the_data(bin_range, df, variable):
    bins = pd.cut(df[variable], bin_range)
    return df.groupby(bins)[variable].size()


def kernel_fisher(dN, N, over, df, variable):
    # k = get_uni_kde(df[variable])
    x = range(0, 300000, 100)
    fi = []
    for i in range(0, N - dN, over):
        temp = df[variable][i : i + dN]
        k = get_uni_kde(temp)
        sorted_w = list(temp.sort_values(axis=0))
        vals = get_kde_probs(k, sorted_w)
        if i % 100 == 0:
            plt.plot(x, k.evaluate(x))
            plt.show()
        fi.append(sum([(vals[x + 1] - vals[x]) ** 2 / vals[x] for x in range(dN)]))
    return fi


def kern(dN, N, over, df, variable):
    # k = get_uni_kde(df[variable])
    x = np.linspace(0, 100, 3000)
    fi = []
    for i in range(0, N - dN, over):
        temp = df[variable][i : i + dN]
        k = get_uni_kde(temp)
        vals = k.pdf(x)
        p_prime2 = np.gradient(np.gradient(vals))
        fi.append(abs(simps(p_prime2 / vals)))
    return fi


def discrete_fisher(number_bins, dN, N, over, x):
    dis_fi = []
    for i in range(0, N - dN, over):
        temp = x[i : i + dN]
        hist = np.histogram(temp, bins=number_bins, density=False)
        counts = list(hist[0] / dN) + [0]
        dis_fi.append(sum([(counts[x + 1] - counts[x]) ** 2 / counts[x] for x in range(number_bins) if counts[x] != 0]))
    return dis_fi


def plot_kern_and_discrete(fig_name, number_bins, dN, N, over, x, kern, disc):
    fig, ax1 = plt.subplots(figsize=(17, 9))
    fig.subplots_adjust(right=0.85)

    lns1, = ax1.plot(x, "b", label="x(t) - Storage")
    ax1.set_ylabel("x(t)")
    ax1.set_xlabel("Time")
    ax1.set_title(f"{fig_name} (dN: {dN}, bins: {number_bins})")

    ax2 = ax1.twinx()
    lns2, = ax2.plot(range(dN, N, over), kern, "r--", label="Kernel FI")
    ax2.set_ylabel("Fisher Info - Kernel")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.05))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    lns3, = ax3.plot(range(dN, N, over), disc, "g--", label="Discrete FI")
    ax3.set_ylabel("Fisher Info - Discrete")

    ax1.yaxis.label.set_color(lns1.get_color())
    ax2.yaxis.label.set_color(lns2.get_color())
    ax3.yaxis.label.set_color(lns3.get_color())

    lns = [lns1, lns2, lns3]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    fig.canvas.set_window_title(f"{fig_name}")
    # plt.savefig(f"PICS/{fig_name}_dN_{dN}_bins_{number_bins}.png")
    plt.show()
    plt.close("all")  # remove plot from memory


def plot_kern_and_diff(fig_name, number_bins, dN, N, over, x, kern, x_range, diff):
    fig, ax1 = plt.subplots(figsize=(17, 9))
    fig.subplots_adjust(right=0.85)

    lns1, = ax1.plot(x, "b", label="x(t) - Storage")
    ax1.set_ylabel("x(t)")
    ax1.set_xlabel("Time")
    ax1.set_title(f"{fig_name} (dN: {dN}, bins: {number_bins})")

    ax2 = ax1.twinx()
    lns2, = ax2.plot(range(dN, N, over), kern, "r--", label="Kernel FI")
    ax2.set_ylabel("Fisher Info - Kernel")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.05))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    lns3, = ax3.plot(x_range, diff, "g--", label=f"Diff{x_range[0]-dN}")
    ax3.set_ylabel("Diff")

    ax1.yaxis.label.set_color(lns1.get_color())
    ax2.yaxis.label.set_color(lns2.get_color())
    ax3.yaxis.label.set_color(lns3.get_color())

    lns = [lns1, lns2, lns3]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    fig.canvas.set_window_title(f"{fig_name}")
    #plt.savefig(f"PICS/{fig_name}_dN_{dN}_bins_{number_bins}_diff.png")
    plt.show()
    plt.close("all")  # remove plot from memory

def list_csv_files(search_dir, reg_to_match):
    """
    This function takes a directory and regex pattern, returns
    all files within the directory that match the regex

    args: 
        search_dir: the directory to be searched
        reg_to_match: the regex pattern to be matched
    returns:
        list: full list of files without the extension
    """
    fnames = os.listdir(search_dir)
    return [fname for fname in fnames if reg_to_match.match(fname)]

def size_of_state(k, data_num, window_size):
    sos_temp = []
    for i in range(len(data_num) - window_size):
        A = data_num[i : i + window_size]
        sos_temp.append(np.std(A, ddof=1))
    if not sos_temp:
        sos = 0
    else:
        sos = min(sos_temp) * k
    return sos

def make_bin_edges(sos, k, first, minim, maxim):
    edges = []
    d_x = sos * k
    first_low, first_high = first - d_x/2, first + d_x/2
    edges.extend([first_high, first_low])
    temp = first_low
    while temp > minim:
        temp -= d_x
        edges.append(temp)
    temp = first_high
    while temp < maxim:
        temp += d_x
        edges.append(temp)
    return sorted(edges)

def discrete_sos_fisher(sos, dN, N, over, x):
    minim, maxim = min(x), max(x)
    bin_edges = make_bin_edges(sos, 2, x[0], minim, maxim)
    dis_fi = []
    for i in range(0, N - dN, over):
        temp = x[i : i + dN]
        hist = np.histogram(temp, bins=bin_edges, density=False)
        counts = list(hist[0] / dN) + [0]
        dis_fi.append(4 * sum([(math.sqrt(counts[x + 1]) - math.sqrt(counts[x])) ** 2 for x in range(len(hist[0]))]))
    return dis_fi

def main(dN, number_bins, filename, directory):
    dN = 47
    directory = '.'
    filename = 'cantar2019.csv'
    # filename = 'res-mgmt-1-s-0.8-0.csv'
    fig_name = filename[:-4]

    df = get_dataframe(directory + "/" + filename)
    # df = pd.concat([historical, df], join='inner', ignore_index=True)
    variable = "storage"
    print(df.shape)

    N = len(df.index)
    over = 1

    x = df[variable]
    if fig_name != 'cantar2019':
        x /= 1000

    sos = size_of_state(2, list(x), dN)
    # kern_fi = kernel_fisher(dN, N, over, df, variable)
    kern_fi = kernel_concise(x, dN, N, over, 10**-9)
    disc_fi = discrete_sos_fisher(sos, dN, N, over, x)
    # disc_fi = discrete_fisher(5, dN, N, over, x)

    n = 1
    x_range = range(dN+n, N, over)


    plot_kern_and_discrete(fig_name, number_bins, dN, N, over, x, kern_fi, disc_fi)
    # plot_kern_and_diff(fig_name, number_bins, dN, N, over, x, kern_fi, x_range, diff)


if __name__ == "__main__":
    directory = "./5-jul-2019"
    reg = re.compile(r"res-mgmt-\d-s-\d.\d-\d+.csv")
    file_list = list_csv_files(directory, reg)
    file_list = [file_list[2]]

    for filename in file_list:
        main(100, 5, filename, directory)
