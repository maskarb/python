import numpy as np

def size_of_state(k, x, window_size):
    sos_temp = []
    for i in range(len(x) - window_size):
        A = x[i : i + window_size]
        sos_temp.append(np.std(A, ddof=1))
    if not sos_temp:
        sos = 0
    else:
        sos = min(sos_temp) * k
    return sos

def find_sos(k, x, window_size):
    sos = []
    for i in range(x.shape[1]):
        sos.append(size_of_state(k, x[:, i], window_size))
    return sos

def make_bin_edges(sos, k, x):
    middle, minim, maxim = np.mean(x), min(x), max(x)
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

def amp_sos_fisher(x_list, k, sos):
    bins = []
    for i in range(x_list.shape[1]):
        bins.append(make_bin_edges(sos[i], k, x_list[:, i]))
    hist = np.histogramdd(x_list, bins=bins)
    counts = [0] + list(hist[0] / len(x_list)) + [0]
    return sum(
        [
            (np.sqrt(counts[x + 1]) - np.sqrt(counts[x])) ** 2
            for x in range(len(hist[0]))
        ]
    )

def temporal(x_list, k, window_size, over):
    sos = find_sos(k, x_list, window_size)
    N = x_list.shape[0]
    fi = []
    for i in range(0, N-window_size, over):
        temp = x_list[i : i+window_size, :]
        fi.append(amp_sos_fisher(temp, k, sos))
    return fi

if __name__ == "__main__":
    r = np.random.randn(100,2)
    temporal(r, 2, 50, 1)