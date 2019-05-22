from ctypes import CDLL
import itertools
import math
import random
import statistics as stat

import numpy as np
from scipy.optimize import least_squares as ls
from scipy.stats import gaussian_kde as gk

import matplotlib.pyplot as plt

from fast_deriv import FastUnivariateDensityDerivative as FUDD

from brazil_percent import ral as inflow

random.seed(100)

# N - number of source points
# X - 1 x N matrix of N source points
# eps - accuracy parameter

# h - optimal bandwidth estimated


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

    sigma = stat.stdev(X_shifted_scale)

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
        verbose=1,
        args=(N, X_shifted_scale, constant1, constant2, eps),
    )

    h = float(h.x) / scale
    return h

def fim(x_list, h, kappa2):
    return 1 / (stat.variance(x_list) + h ** 2 * kappa2)

eps = 10 ** -2

dist = random.gauss
N = 6000





s = [i/10 for i in range(1, 50)]
theoretical = [1/i ** 2 for i in s]

X = [[dist(0, i) for _ in range(N)] for i in s]
print("x created")
calculated_h = [find_opt_h(x, eps) for x in X]
calc_fim = [fim(x, calculated_h[i], 1) for i, x in enumerate(X)]

plt.semilogy(theoretical)
plt.semilogy(calc_fim, 'o')
plt.show()
