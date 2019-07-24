import matplotlib.pyplot as plt
import pandas as pd
import re
import os

from discrete_funcs import temporal_amp, temporal_amp_wind
from kernel_functions import temporal_kern
from fi import fisher

def list_csv_files(search_dir, reg_to_match):
    fnames = os.listdir(search_dir)
    return [fname for fname in fnames if reg_to_match.match(fname)]

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def get_dataframe(filename):
    return pd.read_csv(filename)

def plot_3_fisher(x, fi1, fi2, fi3, dN, over, fig_name):
    N = len(x)
    fi_x = range(dN, 1+N, over)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(right=0.8)

    lns1, = ax1.plot(x, "k", label="x(t) - Storage")
    ax1.set_ylabel("x(t)")
    ax1.set_xlabel("Time (months)")

    ax2 = ax1.twinx()
    lns2, = ax2.plot(fi_x, fi1, "b", label="Kernel FI")
    ax2.set_ylabel("FI - Kernel")
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    lns3, = ax3.plot(fi_x, fi2, "r", label=f"Discrete FI - overlap")
    ax3.set_ylabel("FI - discrete overlap")

    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(ax4)
    ax4.spines["right"].set_visible(True)
    lns4, = ax4.plot(fi_x, fi3, "g", label=f"Discrete FI - disjoint")
    ax4.set_ylabel("FI - discrete disjoint")

    ax1.yaxis.label.set_color(lns1.get_color())
    ax2.yaxis.label.set_color(lns2.get_color())
    ax3.yaxis.label.set_color(lns3.get_color())
    ax4.yaxis.label.set_color(lns4.get_color())

    #lns = [lns1, lns2, lns3, lns4]
    #labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs)
    fig.canvas.set_window_title(f"{fig_name}")
    #plt.savefig(f"PICS/{fig_name}_dN_{dN}_bins_{number_bins}_diff.png")
    plt.show()
    plt.close("all")  # remove plot from memory


def plot_2_kern_amp(x, fi1, fi2, dN, over, fig_name):
    N = len(x)
    fi_x = range(dN, 1+N, over)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(right=0.8)

    lns1, = ax1.plot(x, "k", label="x(t) - Storage")
    ax1.set_ylabel("x(t)")
    ax1.set_xlabel("Time (months)")

    ax2 = ax1.twinx()
    lns2, = ax2.plot(fi_x, fi1, "b", label="Kernel FI")
    ax2.set_ylabel("FI - Kernel")
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    make_patch_spines_invisible(ax3)
    ax3.spines["right"].set_visible(True)
    lns3, = ax3.plot(fi_x, fi2, "g", label=f"Discrete FI - disjoint")
    ax3.set_ylabel("FI - discrete disjoint")

    ax1.yaxis.label.set_color(lns1.get_color())
    ax2.yaxis.label.set_color(lns2.get_color())
    ax3.yaxis.label.set_color(lns3.get_color())

    #lns = [lns1, lns2, lns3, lns4]
    #labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs)
    fig.canvas.set_window_title(f"{fig_name}")
    #plt.savefig(f"PICS/{fig_name}_dN_{dN}_bins_{number_bins}_diff.png")
    plt.show()
    plt.close("all")  # remove plot from memory

def main(dN, k, filename, directory):
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
        dN = 100

    kernel = temporal_kern(x, dN, over, 10 ** -9)
    disjoint_whole = temporal_amp(x, dN, over, k)
    # disjoint_wind = temporal_amp_wind(x, dN, over, k)
    # overlap = fisher(x, dN, over, k)

    plot_2_kern_amp(x, kernel, disjoint_whole, dN, over, fig_name)
    # plot_3_fisher(x, kernel, overlap, disjoint_whole, dN, over, fig_name)


if __name__ == "__main__":
    #directory = "./5-jul-2019"
    #reg = re.compile(r"res-mgmt-\d-s-\d.\d-\d+.csv")
    #file_list = list_csv_files(directory, reg)
    #file_list = [file_list[2]]
    file_list = ['cantar2019.csv']
    directory = '.'

    for filename in file_list:
        main(48, 2, filename, directory)
