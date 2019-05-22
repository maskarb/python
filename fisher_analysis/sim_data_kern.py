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

a = [None] * N
for i in range(N):
    if i < 200:
        a[i] = 0.02
    elif i < 210:
        a[i] = 0.02 + ((i - 200) / 10) * 0.06
    elif i < 500:
        a[i] = 0.08
    elif i < 1800:
        a[i] = 0.08 + ((i - 500) / 1300) * 0.06
        print(a[i])
    else:
        a[i] = 0.14
    

b = 0.58
p = 0.99

z = [1]

x = [0]
for i in range(N):
    z.append(p * z[i] + dist(0, 0.0002))
    x.append(a[i] * math.exp(1) - b * x[i] + (x[i] ** 2 / (1 + x[i] ** 2)))
print("x created")


calculated_h = find_opt_h(a, eps)
calc_fim = [fim(x[i : i + dN], calculated_h, 1) for i in range(0, N - dN, 50)]

fig, ax1 = plt.subplots()

ax1.plot(range(dN, N, 50), calc_fim, "o")

ax2 = ax1.twinx()
ax2.plot(x)

plt.show()
