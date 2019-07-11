import math
import random
from statistics import stdev, variance

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy.misc import derivative
from scipy.optimize import least_squares as ls
from scipy.stats import gaussian_kde as kde
from sklearn.preprocessing import MinMaxScaler

# from brazil_percent import ral as inflow
from fast_deriv import FastUnivariateDensityDerivative as FUDD

# from FastUnivariateDensityDerivative_bak import UnivariateDensityDerivative as FUDD

# N - number of source points
# X - 1 x N matrix of N source points
# eps - accuracy parameter
# h - optimal bandwidth estimated

INF = float("inf")
min_max = MinMaxScaler()

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


def get_kde_probs(kde, datapoints):
    datapoints.append(INF)
    return [kde.integrate_box_1d(-INF, d) for d in datapoints]


def a_different_kern(x_list, i):
    minim, maxim = get_min_max(x_list)
    x = list(np.linspace(minim/0.5, maxim/0.5, 1000))
    k = get_uni_kde(x_list)
    # sorted_w = sorted(x_list)
    vals = get_kde_probs(k, x)
    return sum([(vals[x + 1] - vals[x]) ** 2 / vals[x] for x in range(len(x)-1)])


def yet_another(x_list, h):
    N = 1000
    X = np.linspace(-3, 3, N)
    shift = min(X)
    X_shifted = [x - shift for x in X]

    scale = 1 / max(X_shifted)
    X_shifted_scale = [x * scale for x in X_shifted]
    F2 = FUDD(N, N, X_shifted_scale, X_shifted_scale, h, 2, eps)
    F2.evaluate()
    F = FUDD(N, N, X_shifted_scale, X_shifted_scale, h, 0, eps)
    F.evaluate()
    p2 = np.array(F2.pD)
    p = np.array(F.pD)
    ints = simps(p2 / p)
    return ints

def kernel_again(x_list, i, h):
    minim, maxim = get_min_max(x_list)
    x = np.linspace(minim/0.005, maxim/0.005, 1000)
    norm = min_max.fit_transform(x.reshape(-1, 1))
    k = get_uni_kde(x_list, h)
    vals = k.pdf(norm.reshape(norm.shape[0]))
    #plt.plot(x, vals)
    #plt.show()
    if 0 in vals:
        print(f"ZERO IN VALS! : {i}")
    p_prime = np.gradient(vals)
    p_prime2 = np.gradient(p_prime)
    ints = simps(p_prime2 / vals)
    if ints < 0:
        print("LESS THAN ZERO")
    return ints

def discrete_fisher(x_list, number_bins):
    # minim, maxim = get_min_max(df[variable])
    # bin_range = get_bin_range(5, minim, maxim)
    hist = np.histogram(x_list, bins=number_bins, density=False)
    counts = list(hist[0] / len(x_list)) + [0]
    return sum([(counts[x + 1] - counts[x]) ** 2 / counts[x] for x in range(number_bins) if counts[x] != 0])


dist = random.gauss
eps = 10 ** -9
N = 100

s = [i / 10 for i in range(1, 5000, 100)]
theoretical = [1 / i ** 2 for i in s]

X = [[dist(0, i) for _ in range(N)] for i in s]
print("x created")
calculated_h = [find_opt_h(x, eps) for x in X]
print("h's calculated")
calc_fim = [kernel_again(x, i, calculated_h[i]) for i, x in enumerate(X)]
#calc_fim = [yet_another(x, calculated_h[i]) for i, x in enumerate(X)]
#calc_fim = [a_different_kern(x, i) for i, x in enumerate(X)]
#calc_fim = [discrete_fisher(x, 5) for i, x in enumerate(X)]

plt.semilogy(theoretical)
plt.semilogy(calc_fim, "o")
plt.title('kernel')
plt.show()
