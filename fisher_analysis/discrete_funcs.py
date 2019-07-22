import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_bin_edges(sos, k, x):
    middle, minim, maxim = x[0], min(x), max(x)
    d_x = sos * k
    middle_low_edge, middle_high_edge = middle - d_x/2, middle + d_x/2
    edges = [middle_low_edge, middle_high_edge]
    temp = middle_low_edge
    while temp > minim:
        temp -= d_x
        edges.append(temp)
    temp = middle_high_edge
    while temp < maxim:
        temp += d_x
        edges.append(temp)
    return sorted(edges)

def size_of_state(x, k, window_size):
    sos_temp = []
    for i in range(1 + len(x) - window_size):
        A = x[i : i + window_size]
        sos_temp.append(np.std(A, ddof=1))
    if not sos_temp:
        sos = 0
    else:
        sos = min(sos_temp) * k
    return sos

def amp_sos_fisher(x_list, bins):
    hist = np.histogram(x_list, bins=bins, density=False)
    counts = [0] + list(hist[0] / len(x_list)) + [0]
    return sum(
        [
            (np.sqrt(counts[x + 1]) - np.sqrt(counts[x])) ** 2
            for x in range(len(counts) - 1)
        ]
    )

def temporal_amp(x_list, k, window_size, over):
    N = len(x_list)
    sos = size_of_state(x_list, k, window_size)
    bins = make_bin_edges(sos, k, x_list) ########### move this below
    fi = []
    for i  in range(0, N - window_size, over):
        temp = x_list[i : i + window_size]
        ################## move above to here
        fi.append(amp_sos_fisher(temp, bins))
    return fi

if __name__ == "__main__":
    x = []
    for _ in range(3):
        for _ in range(100):
            x.append(random.gauss(50, 20))
        for _ in range(100):
            x.append(random.gauss(500, 20))

    df = pd.read_csv('cantar2019.csv')
    x = list(df['storage'])

    k = 2
    dN = 48
    over = 1
    fi = temporal_amp(x, k, dN, over)

    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(x, "k")
    ax2 = ax1.twinx()
    ax2.plot(range(dN, len(x), over), fi, "b")
    plt.show()
