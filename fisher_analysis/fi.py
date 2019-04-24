import numpy as np
import pandas as pd

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


def fisher(data_num, Time, w_size, w_incr, sost, f_name):
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
    return FI_final