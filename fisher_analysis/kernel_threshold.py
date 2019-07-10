import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as kde

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
            plt.plot(k.evaluate(x))
            plt.show()
        fi.append(sum([(vals[x + 1] - vals[x]) ** 2 / vals[x] for x in range(dN)]))
    return fi


def discrete_fisher(number_bins, dN, N, over, df, variable):
    # minim, maxim = get_min_max(df[variable])
    # bin_range = get_bin_range(5, minim, maxim)
    dis_fi = []
    for i in range(0, N - dN, over):
        temp = df.iloc[i : i + dN, :][variable]
        hist = np.histogram(temp, bins=number_bins, density=False)
        counts = list(hist[0] / dN) + [0]
        if i % 100 == 0:
            plt.hist(temp, bins=number_bins)
            plt.show()
        # print(f"{i:3d} : {counts}")
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
    #plt.savefig(f"PICS/{fig_name}_dN_{dN}_bins_{number_bins}.png")
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


def main(dN, number_bins, filename, directory):
    # dN = 50
    # directory = '.'
    # filename = 'cantar2019.csv'
    # filename = 'res-mgmt-1-s-0.8-0.csv'
    fig_name = filename[:-4]

    df = get_dataframe(directory + "/" + filename)
    # df = pd.concat([historical, df], join='inner', ignore_index=True)
    variable = "storage"
    print(df.shape)

    N = len(df.index)
    over = 1

    x = df[variable]
    kern_fi = kernel_fisher(dN, N, over, df, variable)
    disc_fi = discrete_fisher(number_bins, dN, N, over, df, variable)

    n = 1
    x_range = range(dN+n, N, over)
    diff = np.diff(kern_fi, n=n)

    plot_kern_and_discrete(fig_name, number_bins, dN, N, over, x, kern_fi, disc_fi)
    #plot_kern_and_diff(fig_name, number_bins, dN, N, over, x, kern_fi, x_range, diff)


if __name__ == "__main__":
    directory = "./5-jul-2019"
    reg = re.compile(r"res-mgmt-\d-s-\d.\d-\d+.csv")
    file_list = list_csv_files(directory, reg)
    file_list = [file_list[0]]

    dN_range = [100]
    bins_range = [50]
    for filename in file_list:
        for dN in dN_range:
            for number_bins in bins_range:
                print(f"dN: {dN}, bin: {number_bins}")
                main(dN, number_bins, filename, directory)
