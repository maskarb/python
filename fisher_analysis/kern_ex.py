from scipy.stats import gaussian_kde as gk
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

import numpy as np
import random

from brazil_percent import dados as inflow

dist = random.gauss
X = [0.75*norm(0.3, 1).pdf(i) + 0.25*uniform(-3, 1).pdf(i) for i in np.linspace(-5, 10, 1000)]
n = len(X)

x_sort = sorted(X)
kernel = gk(x_sort, bw_method='silverman')
points = np.linspace(min(x_sort), max(x_sort), 1000)
fit = kernel(points)
print(x_sort)
h = kernel.factor

print(h)
print(kernel.covariance)
print(1/h)

plt.plot(points, fit, 'k')
plt.show()


