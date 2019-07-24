import matplotlib.pyplot as plt

from numpy import gradient, inf, linspace, pi, power, sqrt, std
from pandas import read_csv
from scipy.integrate import romb, simps
from scipy.optimize import least_squares as ls
from scipy.stats import gaussian_kde as kde

from fast_deriv import FastUnivariateDensityDerivative as FUDD


def _fast_h_fun(h, N, X, c1, c2, eps):
    lam = c2 * h ** (5 / 7)
    D4 = FUDD(N, N, X, X, float(lam), 4, eps)
    D4.evaluate()
    phi4 = sum(D4.pD) / (N - 1)
    return h - c1 * phi4 ** (-1 / 5)


def _get_scale_list(x_list):
    shift = min(x_list)
    scale = 1 / (max(x_list) - shift)
    X_shifted_scale = [(x - shift) * scale for x in x_list]
    return X_shifted_scale, scale


def _get_opt_h(x_list, eps):
    N = len(x_list)
    X_shifted_scale, scale = _get_scale_list(x_list)

    sigma = std(X_shifted_scale)

    phi6 = (-15 / (16 * sqrt(pi))) * power(sigma, -7)
    phi8 = (105 / (32 * sqrt(pi))) * power(sigma, -9)

    g1 = (-6 / (sqrt(2 * pi) * phi6 * N)) ** (1 / 7)
    g2 = (30 / (sqrt(2 * pi) * phi8 * N)) ** (1 / 9)

    D4 = FUDD(N, N, X_shifted_scale, X_shifted_scale, g1, 4, eps)
    D6 = FUDD(N, N, X_shifted_scale, X_shifted_scale, g2, 6, eps)

    D4.evaluate()
    D6.evaluate()

    phi4 = sum(D4.pD) / (N - 1)
    phi6 = sum(D6.pD) / (N - 1)

    constant1 = (1 / (2 * sqrt(pi) * N)) ** (1 / 5)
    constant2 = (-6 * sqrt(2) * phi4 / phi6) ** (1 / 7)

    h_initial = constant1 * phi4 ** (-1 / 5)
    h = ls(
        _fast_h_fun,
        h_initial,
        bounds=(0, inf),
        ftol=1e-14,
        xtol=1e-14,
        verbose=0,
        args=(N, X_shifted_scale, constant1, constant2, eps),
    )
    h = float(h.x) / scale
    return h


def _get_uni_kde(data, bw_method="silverman"):
    return kde(data, bw_method=bw_method)

def _get_array_bounds(kernel, low_bound, high_bound):
    low = ls(_ls_get_value, 0, args=(kernel, low_bound))
    high = ls(_ls_get_value, 0, args=(kernel, high_bound))
    return float(low.x), float(high.x)


def _ls_get_value(x, k, p):
    return p - k.integrate_box_1d(-inf, x)


def kernel_fi(x_list, opt_h):
    kernel = _get_uni_kde(x_list, opt_h)
    low, high = _get_array_bounds(kernel, 0.0001, 0.9999)
    x = linspace(low, high, 2 ** 11 + 1)
    probs = kernel.pdf(x)
    p_prime2 = gradient(probs, x) ** 2
    return romb(p_prime2 / probs)


def temporal_kern(x, eps, dN, over):
    opt_h = _get_opt_h(x, eps)
    N = len(x)
    fi = []
    for i in range(0, 1 + N - dN, over):
        window = x[i : i + dN]
        fi.append(kernel_fi(window, opt_h))
    return fi

def main(filename, k, dN, over, eps):
    df = read_csv(filename)
    x = list(df['storage'] / 1000)
    return temporal_kern(x, eps, dN, over), x

if __name__ == "__main__":
    k = 2
    dN = 47
    over = 1
    eps = 10 ** -9
    nums = [2, 8]

    fishers = []
    for i in nums:
        fishers.append(main(f"for_thesis/res-mgmt-0-s-0.{i}-0.csv", k, dN, over, eps))

    for i, fi in enumerate(fishers):
        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        ln1, = ax1.plot(fi[1], color='k', label=f"x = {nums[i]}0%")
        ax2 = ax1.twinx()
        ln2 = ax2.plot(range(dN, 1 + len(fi[1]), over), fi[0], color='b', label="Kernel")
        ax1.set_ylabel("x(t) / 1000")
        ax1.set_xlabel("Time (month)")
        ax2.set_ylabel("Kernel")
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        plt.tight_layout()
        plt.show()