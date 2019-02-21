#%%
import math
import operator as op
from functools import reduce
from statistics import mean, variance

import lmoments3 as lm
from lmoments3 import distr


def m_r(x, r):
    n = len(x)
    return 1 / n * sum([math.pow(i - mean(x), r) for i in x])


def s(x):
    return math.sqrt(m_r(x, 2))


def b1(x):
    return m_r(x, 3) / math.pow(s(x), 3)


def b2(x):
    return m_r(x, 4) / math.pow(s(x), 4)


#%%
def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def lam_r(r, lis):
    n = len(lis)
    return (
        math.pow(r, -1)
        * (math.pow(ncr(n, r), -1))
        * sum([math.pow(-1, (r - j)) * ncr(r - 1, j) * x for j, x in enumerate(lis)])
    )


def l_1(lis):
    n = len(lis)
    return ncr(n, 1) ** -1 * sum(lis)


def l_2(lis):
    n = len(lis)
    return (
        0.5
        * ncr(n, 2) ** -1
        * sum([(ncr(i - 1, 1) - ncr(n - i, 1)) * x for i, x in enumerate(lis)])
    )

#%%
x = [
    357,
    221,
    -4,
    -52,
    103,
    48,
    723,
    195,
    62,
    325,
    20,
    55,
    1935,
    230,
    371,
    114,
    110,
    281,
    332,
    -66,
    973,
    67,
    141,
    1259,
    47,
    81,
    638,
    130,
    -19,
    25,
    1348,
]
#%%
loc, scale, shape = lm.lmom_ratios(x, nmom=3)

from scipy.stats import pearson3
import matplotlib.pyplot as plt
import numpy as np

p = pearson3(loc=loc, skew=shape, scale=scale)

x = sorted(x)
vals1 = np.linspace(0, 1, 31)
vals = np.linspace(0, 1, 1000)
ys = p.ppf(vals)
plt.plot(vals, ys, vals1, x)
plt.show()