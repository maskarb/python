import math
from statistics import stdev, variance
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares as ls

from brazil_percent import dados as w
from fast_deriv import FastUnivariateDensityDerivative as FUDD
from res_workflow import size_of_state, fisher

# from FastUnivariateDensityDerivative import UnivariateDensityDerivative as FUDD
# from FastUnivariateDensityDerivative_bak import UnivariateDensityDerivative as FUDD

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
def sinu(alpha, beta, t):
    return (alpha + beta*t) * math.sin(2*math.pi*t)
def convert_to_list_of_list(lis):
    return [[i] for i in lis]
def give_what_i_need(lis):
    return [i[-1] for i in lis], [i[-2] for i in lis]

start = time.time()

eps = 10 ** -6
N = len(w)
dN = 28
over = 1


calculated_h = find_opt_h(w, eps)
calc_fimk = [fim(w[i : i + dN], calculated_h, 1) for i in range(0, N - dN, over)]
calc_fimd = fisher(convert_to_list_of_list(w), range(N), dN, over, size_of_state(w, dN), "sim_data")

end = time.time()
print(f'Total time (s): {end - start}')

fig, ax1 = plt.subplots(figsize=(17, 9))

lns1 = ax1.plot(range(dN, N, over), calc_fimk, "b:.", label="kernel FI")
ax1.set_xlabel("Time")
ax1.set_ylabel("Fisher Info")
ax1.set_title("Kernel Method")

ax2 = ax1.twinx()
lns2 = ax2.plot(w, "k", label="x(t)")
ax2.set_ylabel("x(t)")
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)
plt.savefig(f"PICS/dados_fik_{dN}_{over}.png")
plt.close("all")  # remove plot from memory


fig, ax1 = plt.subplots(figsize=(17, 9))

x1, y1 = give_what_i_need(calc_fimd)
lns1 = ax1.plot(x1, y1, "b:.", label="discrete FI")
ax1.set_xlabel("Time")
ax1.set_ylabel("Fisher Info")
ax1.set_title("Discrete Method")

ax2 = ax1.twinx()
lns2 = ax2.plot(w, "k", label="x(t)")
ax2.set_ylabel("x(t)")
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)
plt.savefig(f"PICS/dados_fid_{dN}_{over}.png")
plt.close("all")  # remove plot from memory