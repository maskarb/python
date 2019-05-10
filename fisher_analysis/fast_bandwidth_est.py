from ctypes import CDLL
import itertools
import math
import random
import statistics as stat

import numpy as np
from scipy.optimize import least_squares as ls
from scipy.stats import gaussian_kde as gk

import matplotlib.pyplot as plt


from FastUnivariateDensityDerivative import UnivariateDensityDerivative as FUDD

from brazil_percent import ral as inflow

# N - number of source points
# X - 1 x N matrix of N source points
# eps - accuracy parameter

# h - optimal bandwidth estimated


def fast_h_fun(h, N, X, c1, c2, eps):
    lam = c2 * h ** (5 / 7)
    D4 = FUDD(N, N, X, X, lam, 4, eps)
    D4.evaluate()
    phi4 = sum(D4.pD) / (N - 1)
    return h - c1 * phi4 ** (-1 / 5)


eps = 10 ** -2

dist = random.gauss
uni = random.uniform
N = 6000
X = sorted([0.75*dist(0.3, 0.2) + 0.25*uni(-0.3, 0.2) for _ in range(N)])
Y = list(np.linspace(-5, 5, N))

#X = inflow
N = len(X)
print("x created")


shift, yshift = min(X), min(Y)
X_shifted = [x - shift for x in X]
Y_shifted = [y - yshift for y in Y]

scale = 1 / max(X_shifted)
yscale = 1 / max(Y_shifted)
X_shifted_scale = [x * scale for x in X_shifted]
Y_shifted_scale = [y * yscale for y in Y_shifted]

sigma = stat.stdev(X_shifted_scale)

phi6 = (-15 / (16 * math.sqrt(math.pi))) * math.pow(sigma, -7)
phi8 = (105 / (32 * math.sqrt(math.pi))) * math.pow(sigma, -9)

g1 = (-6 / (math.sqrt(2 * math.pi) * phi6 * N)) ** (1 / 7)
g2 = (30 / (math.sqrt(2 * math.pi) * phi8 * N)) ** (1 / 9)

D4 = FUDD(N, N, X_shifted_scale, X_shifted_scale, g1, 4, eps)
D4.evaluate()
phi4 = sum(D4.pD) / (N - 1)
print('phi4')

D6 = FUDD(N, N, X_shifted_scale, X_shifted_scale, g2, 6, eps)
D6.evaluate()
phi6 = sum(D6.pD) / (N - 1)
print('phi6')


constant1 = (1 / (2 * math.sqrt(math.pi) * N)) ** (1 / 5)
constant2 = (-6 * math.sqrt(2) * phi4 / phi6) ** (1 / 7)

h_initial = constant1 * phi4 ** (-1 / 5)

print(h_initial)
x_sort = sorted(X)
kernel = gk(x_sort, bw_method='silverman')
points = np.linspace(min(x_sort), max(x_sort), 1000)
fit = kernel(points)
print(kernel.factor)


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

print(h)

kernel2 = gk(x_sort, bw_method=h)
points2 = np.linspace(min(x_sort), max(x_sort), 1000)
fit2 = kernel2(points2)

plt.plot(points, fit, 'k')
plt.plot(points2, fit2, 'g')

D0 = FUDD(N, N, X_shifted_scale, X_shifted_scale, kernel.factor, 0, eps)
D0.evaluate()
y = [y / scale + shift for y in D0.pD]
plt.plot(x_sort, D0.pD, 'r')
plt.show()