import math
import random
from statistics import mean, stdev, variance

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import romb, simps
from scipy.misc import derivative
from scipy.optimize import least_squares as ls
from scipy.stats import gaussian_kde as kde


# from brazil_percent import ral as inflow
from fast_deriv import FastUnivariateDensityDerivative as FUDD

# from FastUnivariateDensityDerivative_bak import UnivariateDensityDerivative as FUDD

# N - number of source points
# X - 1 x N matrix of N source points
# eps - accuracy parameter
# h - optimal bandwidth estimated

INF = float("inf")


def fast_h_fun(h, N, X, c1, c2, eps):
    lam = c2 * h ** (5 / 7)
    D4 = FUDD(N, N, X, X, float(lam), 4, eps)
    D4.evaluate()
    phi4 = sum(D4.pD) / (N - 1)
    return h - c1 * phi4 ** (-1 / 5)


def find_opt_h(x_list, eps):
    N = len(x_list)
    X = x_list
    shift = min(X)
    X_shifted = [x - shift for x in X]

    scale = 1 / max(X_shifted)
    X_shifted_scale = [x * scale for x in X_shifted]

    sigma = stdev(X_shifted_scale)

    phi6 = (-15 / (16 * math.sqrt(math.pi))) * math.pow(sigma, -7)
    phi8 = (105 / (32 * math.sqrt(math.pi))) * math.pow(sigma, -9)

    g1 = (-6 / (math.sqrt(2 * math.pi) * phi6 * N)) ** (1 / 7)
    g2 = (30 / (math.sqrt(2 * math.pi) * phi8 * N)) ** (1 / 9)

    D4 = FUDD(N, N, X_shifted_scale, X_shifted_scale, g1, 4, eps)
    D4.evaluate()
    phi4 = sum(D4.pD) / (N - 1)

    D6 = FUDD(N, N, X_shifted_scale, X_shifted_scale, g2, 6, eps)
    D6.evaluate()
    phi6 = sum(D6.pD) / (N - 1)

    constant1 = (1 / (2 * math.sqrt(math.pi) * N)) ** (1 / 5)
    constant2 = (-6 * math.sqrt(2) * phi4 / phi6) ** (1 / 7)

    h_initial = constant1 * phi4 ** (-1 / 5)

    h = ls(
        fast_h_fun,
        h_initial,
        bounds=(0, np.inf),
        ftol=1e-14,
        xtol=1e-14,
        verbose=0,
        args=(N, X_shifted_scale, constant1, constant2, eps),
    )

    h = float(h.x) / scale
    return h


def get_min_max(lis):
    return min(lis), max(lis)


def get_uni_kde(data, bw_method="silverman"):
    return kde(data, bw_method=bw_method)


def _ls_get_value(x, k, p):
    return p - k.integrate_box_1d(-np.inf, x)

def get_array_bounds(kernel, low_bound, high_bound):
    low = ls(_ls_get_value, 0, args=(kernel, low_bound))
    high = ls(_ls_get_value, 0, args=(kernel, high_bound))
    return float(low.x), float(high.x)

def kernel_fi(x_list, eps):
    opt_h = find_opt_h(x_list, eps)
    kernel = get_uni_kde(x_list, opt_h)
    low, high = get_array_bounds(kernel, 0.0001, 0.9999)
    x = np.linspace(low, high, 2 ** 11 + 1)
    probs = kernel.pdf(x)
    p_prime2 = np.gradient(probs, x) ** 2
    return romb(p_prime2 / probs)

def temporal_kern(x, dN, over):
    N = len(x)
    fi = []
    for i in range(0, N - dN, over):
        window = x[i : i + dN]
        fi.append(kernel_fi(window, eps))
    return fi

def temporal_amp_disc(x, dN, over, f, k):
    N = len(x)
    sos = size_of_state(k, x, dN)
    fi = []
    for i in range(0, N - dN, over):
        window = x[i : i + dN]
        fi.append(f(window, sos))
    return fi


def temporal_prob_disc(x, dN, over, f, b):
    N = len(x)
    fi = []
    for i in range(0, N - dN, over):
        window = x[i : i + dN]
        fi.append(f(window, b))
    return fi


def discrete_fisher(x_list, number_bins):
    hist = np.histogram(x_list, bins=number_bins, density=False)
    counts = list(hist[0] / len(x_list)) + [0]
    return sum(
        [
            (counts[x + 1] - counts[x]) ** 2 / counts[x]
            for x in range(number_bins)
            if counts[x] != 0
        ]
    )

def amp_fisher(x_list, number_bins):
    hist = np.histogram(x_list, bins=number_bins, density=False)
    counts = [0] + list(hist[0] / len(x_list)) + [0]
    return sum(
        [
            (math.sqrt(counts[x + 1]) - math.sqrt(counts[x])) ** 2
            for x in range(number_bins+1)
        ]
    )

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

def amp_sos_fisher(x_list, k):
    sos = np.std(x_list, ddof=1) * k
    bins = make_bin_edges(sos, k, x_list[0], min(x_list), max(x_list))
    hist = np.histogram(x_list, bins=bins, density=False)
    counts = [0] + list(hist[0] / len(x_list)) + [0]
    return sum(
        [
            (math.sqrt(counts[x + 1]) - math.sqrt(counts[x])) ** 2
            for x in range(len(hist[0]))
        ]
    )

def prob_sos_fisher(x_list, k):
    sos = np.std(x_list, ddof=1) * k
    bins = make_bin_edges(sos, k, x_list[0], min(x_list), max(x_list))
    hist = np.histogram(x_list, bins=bins, density=False)
    counts = [0] + list(hist[0] / len(x_list)) + [0]
    return sum(
        [
            (counts[x + 1] - counts[x]) ** 2 / counts[x]
            for x in range(len(hist[0]))
            if counts[x] != 0
        ]
    )


dist = np.random.normal
eps = 10 ** -9

s = [i / 10 for i in range(100, 5000, 100)]
theoretical = [1 / i ** 2 for i in s]

xes = []
for _ in range(3):
    for _ in range(100):
        xes.append(dist(0, 20))
    for _ in range(100):
        xes.append(dist(200, 20))


N = len(xes)
dN = 50
over = 1

#for n in N:
#    xes.append([[[dist(0, i) for _ in range(n)] for _ in range(30)] for i in s])

#xes = [[dist(0, i) for _ in range(30)] for i in s]
#V = [[dist(0, i) for _ in range(1000)] for i in s]

print("x created")
# calculated_h = [find_opt_h(x, eps) for x in X]
# print("h's calculated")
calc_fims = temporal_kern(xes, dN, over)

fig, ax1 = plt.subplots(figsize=(5, 4))
#ax1.set_yscale("log", nonposy='mask')
ax1.plot(xes, "k")
ax2 = ax1.twinx()
ax2.plot(list(range(dN, N, over)), calc_fims)
#plt.semilogy(s, calc_fim2, "-o", label="N = 40")
#plt.semilogy(s, calc_fim3, "-o", label="N = 100")
#plt.semilogy(s, calc_fim4, "-o", label="N = 1000")
#plt.semilogy(s, calc_fim, "-o", label="N = 100")
#plt.title("Amplitude Discrete FI method - $\Delta s = k\sigma$")
plt.title("Kernel FI method - Temporal Series")

plt.xlabel("Time (t)")
plt.ylabel("I")
plt.show()
