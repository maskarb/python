import math
import random
from statistics import stdev, variance

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares as ls

from brazil_percent import ral as inflow
from fast_deriv import FastUnivariateDensityDerivative as FUDD

# from FastUnivariateDensityDerivative import UnivariateDensityDerivative as FUDD

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
        verbose=1,
        args=(N, X_shifted_scale, constant1, constant2, eps),
    )

    h = float(h.x) / scale
    return h


def fim(x_list, h, kappa2):
    return 1 / (variance(x_list) + h ** 2 * kappa2)


eps = 10 ** -6
N = 2000
dN = 100
dist = random.gauss


def get_a(t):
    assert t >= 0
    if t < 200:
        a = 0.02
    elif 200 <= t < 210:
        a = 0.02 + ((t - 200) / 10) * 0.06
    elif 210 <= t < 500:
        a = 0.08
    elif 500 <= t < 1800:
        a = 0.08 + ((t - 500) / 1300) * 0.06
    else:
        a = 0.14
    return a


b = 0.58
p = 0.99

z = [0.0] * (N + 1)

x = [0] * (N + 1)
for i in range(N):
    z[i + 1] = (p * z[i] + dist(0, math.sqrt(0.0002)))
    x[i + 1] = ( (get_a(i) * math.exp(z[i])) - (b * x[i]) + ( (x[i] ** 2) / (1 + (x[i] ** 2) )))
print("x created")


calculated_h = find_opt_h(x, eps)
calc_fim = [fim(x[i : i + dN], calculated_h, 1) for i in range(0, N - dN, 50)]

fig, ax1 = plt.subplots()

ax1.plot(range(dN, N, 50), calc_fim, "o")

ax2 = ax1.twinx()
ax2.plot(x)

plt.show()
