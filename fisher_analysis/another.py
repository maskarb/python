import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.stats import gaussian_kde as kde

from brazil_percent import ral08 as w
from fisher_funcs import find_opt_h, fid, fik, size_of_state, convert_to_list_of_list, give_what_i_need

# N - number of source points
# X - 1 x N matrix of N source points
# eps - accuracy parameter
# h - optimal bandwidth estimated
########################################################################################################
from scipy import stats
def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2

m1, m2 = measure(2000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
########################################################################################################
df = pd.read_csv('res-s-0.4-0.csv')
w = df[['storage','deficit']].values.T



# w = tuple(w)

start = time.time()

eps = 10 ** -12
_, N = w.shape
dN = 36
over = 1

fi = []
r = 1
start = r-1
end = dN-1
it = range(start, end, r)
for i in range(0, N-dN, over):
    sorted_w = w[:, w[0, i:i+dN].argsort()]
    k = kde(sorted_w, 'silverman')
    probs = k.pdf(sorted_w)
    fi.append(sum([(probs[x] - probs[x+1]) ** 2 / probs[x] for x in it]))

print(len(it))
_, ax1 = plt.subplots()
lns1 = ax1.plot(range(dN, N, over), fi, "r--", label="FI")
ax1.set_xlabel("Time")
ax1.set_ylabel("Fisher Info")
ax1.set_title("Kernel Method")

ax2 = ax1.twinx()
lns2 = ax2.plot(w, "b", label="x(t) - storage")
ax2.set_ylabel("x(t)")
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)
plt.show()