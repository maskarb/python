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
x = (
 816, 1884, 1039,  340, 1919,  784,  340, 1310, 2084, 1174,
1868,  784,  843, 1682, 1151, 3006, 1598, 1127,  160,  568,
 658,  240,  857,  368, 1082,  101,  736, 1546,  110,  155,
 854,
)

y = (
2394, 2558, 2497,  453, 1908,  607, 2218, 2029,  347,  575,
 792, 1213, 1406, 1341, 1533, 3393,  480, 1579,  473,  210,
2131,  961,  636,  343,  586,  221,  298, 2200,  237,  275,
 980,
)

import matplotlib.pyplot as plt

plt.plot(x, y, '.')
#%%
l1, l2, l3, l4 = lm.lmom_ratios(x, nmom=4)
t3 = l3/l2
t4 = l4/l2

#%%
from scipy.stats import pearson3, kappa4
import matplotlib.pyplot as plt
import numpy as np

p = kappa4(h=l4, loc=l1, k=l3, scale=l2)

x = sorted(x)
vals1 = np.linspace(0, 1, 31)
vals = np.linspace(0, 1, 1000)
ys = p.ppf(vals)
plt.plot(vals, ys, vals1, x)
plt.show()

#%% GEV
z = 2/(3 + t3) - math.log(2)/math.log(3)
k = 7.8590*z + 2.9554*math.pow(z, 2)
alpha = l2*k / (1 - math.pow(2, -k)) * math.gamma(1+k)
eta = l1 + alpha * (math.gamma(1+k) - 1)/k


#%% GLO
k = -t3
alpha = l2/math.gamma(1+k)*math.gamma(1-k)
eta = l1 + (l2 - alpha)/k