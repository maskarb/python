from scipy.stats import gaussian_kde as gk
import matplotlib.pyplot as plt

import numpy as np

from brazil_percent import dados as inflow

X = inflow

x_sort = sorted(X)
kernel = gk(x_sort, bw_method='silverman')
points = np.linspace(min(x_sort), max(x_sort), 1000)
fit = kernel(points)

print(kernel.factor)

plt.plot(points, fit, 'k')
plt.show()