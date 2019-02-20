import math
import random
import statistics as stat

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf, erfinv
from scipy.stats import pearsonr

from historical_data import evaps, flows, precs


def make_pmf(values):
    """This function takes discrete variables and returns probability mass function."""
    y = [0] + values
    x = list(np.linspace(0, 1, len(y)))
    return InterpolatedUnivariateSpline(x, y)


def map_quantiles(new_dist, old_dist, prob_list):
    """This function takes the new and old dsitributions and quantile maps the given probabilities."""
    new_probs = []
    for p in prob_list:
        temp = new_dist.ppf(p)
        new_probs.append(old_dist.cdf(temp))
    return new_probs


def phi(x, mu, sd):
    """Cumulative distribution function for the normal distribution"""
    return (1 + erf((x - mu) / (sd * math.sqrt(2)))) / 2


def quantile(p, mu, sd):
    """Quantile function for the normal distribution"""
    return mu + sd * math.sqrt(2) * erfinv(2 * p - 1)


def quantile_map(mean, s_mean, sd, prob_list):
    new_probs = []
    for p in prob_list:
        temp = quantile(p, s_mean, sd)
        new_probs.append(phi(temp, mean, sd))
    return new_probs


def make_new_point(values, shift_factor, probs):
    """This function takes a set of values, a shift factor, a set of probabilities and generates a shifted value."""
    values = sorted(values)
    old_ave = stat.mean(values)
    std_dev = stat.stdev(values)
    new_ave = old_ave * shift_factor

    c_spline = make_pmf(values)
    new_probs = quantile_map(old_ave, new_ave, std_dev, probs)
    new_vals = list(c_spline(new_probs))
    n_spline = make_pmf(new_vals)

    sample = random.uniform(0, 1)
    val_for_timepoint = float(n_spline(sample))

    return val_for_timepoint


data = {k: [] for k in range(600)}
shift = 0.4

shiftFactor = list(np.linspace(1, shift, 600))
probs = list(np.linspace(0, 1, 21))

for i, key in enumerate(data.keys()):
    data[key].append(shiftFactor[i])


def memoize(f):
    memo = {}

    def helper(month, val1, val2):
        if month not in memo:
            memo[month] = f(val1, val2)
        return memo[month]

    return helper


def correlation(val1, val2):
    cor, __ = pearsonr(val1, val2)
    return cor


correlation = memoize(correlation)


for i in range(600):
    month = i % 12  # 0 is January
    flow = flows[month]
    precip = precs[month]
    evapor = evaps[month]

    precip_corr = correlation(month, flow, precip)
    evapor_corr = correlation(month, flow, precip)

    if shiftFactor[i - 1] < 1:
        precip_shift = 1 - (1 - shiftFactor[i - 1]) * precip_corr
        evapor_shift = 1 - (1 - shiftFactor[i - 1]) * evapor_corr
    else:
        precip_shift = (shiftFactor[i - 1] - 1) * precip_corr + 1
        evapor_shift = (shiftFactor[i - 1] - 1) * evapor_corr + 1

    flows_recon = make_new_point(flow, shiftFactor[i], probs)
    precip_recon = make_new_point(precip, precip_shift, probs)
    evapor_recon = make_new_point(evapor, evapor_shift, probs)
    data[i].extend([flows_recon, precip_recon, evapor_recon])
