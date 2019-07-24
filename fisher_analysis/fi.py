import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cast_x_list(lis):
    return [[i] for i in lis]
def give_what_i_need(lis):
    return [x[-1] for x in lis]


def size_of_state(data_num, window_size, k):
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
            sos.append(min(sos_temp) * k)
    return sos


def fisher_univariate(x_list, k, w_size, w_incr):
    sost = size_of_state(x_list, w_size, k)
    data_num = cast_x_list(x_list)
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
    return give_what_i_need(FI_final)

def main(filename, k, dN, over, eps):
    df = pd.read_csv(filename)
    x = list(df['storage'] / 1000)
    return fisher_univariate(x, k, dN, over), x

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
        ln2 = ax2.plot(range(dN, 1 + len(fi[1]), over), fi[0], color='r', label="Amplitude - Overlapping")
        ax1.set_ylabel("x(t) / 1000")
        ax1.set_xlabel("Time (month)")
        ax2.set_ylabel("Amplitude - Overlapping")

        plt.tight_layout()
        plt.show()