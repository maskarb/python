from scipy.stats import gaussian_kde as gk
from scipy.stats import norm, uniform
from scipy.special import hermitenorm as hermite

import matplotlib.pyplot as plt

import numpy as np
import random

from brazil_percent import dados as inflow

dist = random.gauss
X = [dist(0, 1) for _ in range(10000)]
n = len(X)

x_new = np.linspace(-5, 5, 1000)
y = norm.pdf(x_new)

x_sort = sorted(X)
kernel = gk(x_sort, bw_method='silverman')
points = np.linspace(min(x_sort), max(x_sort), 1000)
fit = kernel(points)
print(x_sort)
h = kernel.factor

print(h)
print(kernel.covariance)
print(1/h)

y_herm = [hermite(2, True)(i) for i in x_new]

plt.hist(x_sort, 100, density=True)
plt.plot(x_new, y, 'k')
plt.plot(x_new, y_herm)
plt.show()

print(hermite(10))


