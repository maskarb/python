import math
from statistics import stdev, variance

import numpy as np
import pandas as pd
from scipy.optimize import least_squares as ls

from fast_deriv import FastUnivariateDensityDerivative as FUDD


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


def fik(x_list, h, kappa2):
    return len(x_list) / (variance(x_list) + h ** 2 * kappa2)
def sinu(alpha, beta, t):
    return (alpha + beta*t) * math.sin(2*math.pi*t)
def convert_to_list_of_list(lis):
    return [[i] for i in lis]
def give_what_i_need(lis):
    return [i[-1] for i in lis], [i[-2] for i in lis]

def size_of_state(data_num, window_size):
    d_f = pd.DataFrame(data_num)
    sos = []
    for j in range(len(d_f.columns)):
        sos_temp = []
        for i in d_f.index:
            A = list(d_f[j][i:i+window_size])
            A_1 = [float(i) for i in A if i != 0]
            if len(A_1) == window_size:
                sos_temp.append(np.std(A_1, ddof=1))
        if not sos_temp:
            sos.append(0)
        else:
            sos.append(min(sos_temp) * 2)
    return sos

def fid(data_num, Time, w_size, w_incr, sost, f_name):
    FI_final, k_init = [], []
    for i in range(0, len(data_num), w_incr):
        data_win = data_num[i:i+w_size]
        if len(data_win) == w_size:
            _bin = []
            for m in range(w_size):
                bin_temp=[]
                for n in range(w_size):
                    if m==n:
                        bin_temp.append('I')
                    else:
                        bin_temp_1=[]
                        for k in range(len(data_win[n])):
                            if (abs(data_win[m][k] - data_win[n][k])) <= sost[k]:
                                bin_temp_1.append(1)
                            else:
                                bin_temp_1.append(0)
                        bin_temp.append(sum(bin_temp_1))
                _bin.append(bin_temp)
            FI = []
            for tl in range(1, 101):
                tl1 = len(sost) * float(tl) / 100
                bin_1, bin_2 = [], []
                for j, _bin_val in enumerate(_bin):
                    if j not in bin_2:
                        bin_1_temp = [j]
                        for i in range(len(_bin[j])):
                            if _bin_val[i] != 'I' and _bin_val[i] >= tl1 and i not in bin_2:
                                bin_1_temp.append(i)
                        bin_1.append(bin_1_temp)
                        bin_2.extend(bin_1_temp)
                prob = [0]
                for i in bin_1:
                    prob.append(float(len(i))/len(bin_2))
                prob.append(0)
                prob_q = []
                for i in prob:
                    prob_q.append(np.sqrt(i))
                FI_temp = 0
                for i in range(len(prob_q) - 1):
                    FI_temp += (prob_q[i] - prob_q[i+1]) ** 2
                FI_temp = 4 * FI_temp
                FI.append(FI_temp)
            for i in range(len(FI)):
                if FI[i] != 8.0:
                    k_init.append(FI.index(FI[i]))
                    break
            FI_final.append(FI)
    if not k_init:
        k_init.append(0)
    for i, FI_val in enumerate(FI_final): #range(0, len(FI_final)):
        FI_final[i].append(
            float(sum(FI_val[min(k_init):len(FI_val)])) /
            len(FI_val[min(k_init):len(FI_val)])
            )
        FI_final[i].append(Time[(i*w_incr+w_size)-1])
    df_FI = pd.DataFrame(FI_final)
    df_FI.to_csv(f"{f_name}_FI.csv", index=False, header=False)
    print("Fisher Done. CSV")
    return FI_final