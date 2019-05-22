import math
import random
import statistics as stat

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.misc import derivative
from scipy.optimize import least_squares as ls

from brazil_percent import dados as X
from FastUnivariateDensityDerivative import UnivariateDensityDerivative as FUDD


def fast_h_fun(h, N, X, c1, c2, eps):
    lam = c2 * h ** (5 / 7)
    D4 = FUDD(N, N, X, X, lam, 4, eps)
    D4.evaluate()
    phi4 = sum(D4.pD) / (N - 1)
    return h - c1 * phi4 ** (-1 / 5)


def ker(x, x_list, h):
    return sum(
        [(x ** 3 / 3 - c * x ** 2 + c ** 2 * x - h ** 2 * x) / h ** 4 for c in x_list]
    )


eps = 10 ** -2

dist = random.gauss
# X = [dist(0, 0.1) for _ in range(600)]
N = len(X)

shift = min(X)
X_shifted = [x - shift for x in X]

scale = 1 / max(X_shifted)
X_shifted_scale = [x * scale for x in X_shifted]
print(scale)

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
    verbose=2,
    args=(N, X_shifted_scale, constant1, constant2, eps),
)

h = float(h.x) / scale
print(f"BANDWIDTH (h): {h:.5f}")
inf = float("inf")


fim = []
for i in range(0, len(X_shifted_scale) - 36, 1):
    fim.append(ker(X[i + 36], X[i : i + 36], h) / 36)


plt.plot(fim)
plt.show()
