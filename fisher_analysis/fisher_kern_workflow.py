
from ast import literal_eval

import matplotlib.pyplot as plt

from brazil_percent import ral08 as w
import fisher_funcs as ff



def plot_fik(x, calc_fik, dN, N, over, fname):
    fig, ax1 = plt.subplots(figsize=(17, 9))
    lns1 = ax1.plot(range(dN, N, over), calc_fik, "b:.", label="kernel FI")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Fisher Info")
    ax1.set_title("Kernel Method")
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x, "k", label="x(t)")
    ax2.set_ylabel("x(t)")
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    plt.savefig(f"PICS/{fname}-fik-{dN}-{over}.png")
    plt.close("all")  # remove plot from memory


def plot_fid(x, calc_fid, dN, N, over, fname):
    fig, ax1 = plt.subplots(figsize=(17, 9))
    x1, y1 = ff.give_what_i_need(calc_fid)
    lns1 = ax1.plot(x1, y1, "b:.", label="discrete FI")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Fisher Info")
    ax1.set_title("Discrete Method")
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x, "k", label="x(t)")
    ax2.set_ylabel("x(t)")
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    plt.savefig(f"PICS/{fname}-fid-{dN}-{over}.png")
    plt.close("all")  # remove plot from memory


def main_sequence(file_no_ext, dN, over):
    fname_split = file_no_ext.split('-')
    f_name = file_no_ext + '.csv'

    data_num = []

    headers, Data = ff.read_csv_headers(f_name)
    var_list = ['storage', ]
    index_list = ff.get_variable_index(var_list, headers)
    for row in Data:
        data_num.append(literal_eval(row[i])/1000 for i in index_list)


    eps = 10 ** -12
    N = len(data_num)

    calculated_h = ff.find_opt_h(data_num, eps)
    calc_fik = [ff.fik(data_num[i : i + dN], calculated_h, 1) for i in range(0, N - dN, over)]
    calc_fid = ff.fid(ff.convert_to_list_of_list(data_num), range(N), dN, over, ff.size_of_state(data_num, dN), file_no_ext)

    plot_fik(data_num, calc_fik, dN, N, over, file_no_ext)
    plot_fid(data_num, calc_fid, dN, N, over, file_no_ext)


def main():
    file_list = ff.list_csv_files('./')
    for fname in file_list:
        if int(fname.split('-')[4].split('.')[1]) >= 0:
            main_sequence(fname, 96, 1)

    # main_sequence('res-mgmt-0-s-0.2-1', 48, 4, 7)


if __name__ == '__main__':
    main()