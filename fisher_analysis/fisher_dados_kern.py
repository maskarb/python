import time

import matplotlib.pyplot as plt

from brazil_percent import dados as w
from fisher_funcs import find_opt_h, fid, fik, size_of_state, convert_to_list_of_list, give_what_i_need

# N - number of source points
# X - 1 x N matrix of N source points
# eps - accuracy parameter
# h - optimal bandwidth estimated

start = time.time()

eps = 10 ** -6
N = len(w)
dN = 28
over = 1


calculated_h = find_opt_h(w, eps)
calc_fimk = [fik(w[i : i + dN], calculated_h, 1) for i in range(0, N - dN, over)]
calc_fimd = fid(convert_to_list_of_list(w), range(N), dN, over, size_of_state(w, dN), "sim_data")

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