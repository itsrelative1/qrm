import matplotlib.pyplot as plt
import numpy as np

hs_97 = [0, 0.28152, 0.17785, 0.00035, 0.0, 0.16009, 0.00108]
hs_97_mean = 0.08869857142857142
hs_99 = [0, 0, 0.00012, 0, 0, 0.11101, 0]
hs_99_mean = 0.015875714285714285

VC_97_t3 = [0.00673, 0.35994, 0.09421, 0.00035, 2e-05, 0.2036, 0.00108]
VC_97_mean_t3 = 0.09513285714285714
VC_99_t3 = [0, 0.18311, 0.00012, 0, 0, 0.11101, 0]
VC_99_mean_t3 = 0.04203428571428571

# VC_97_n = [0.04137, 0.1767, 0.04137, 0.01073, 0.09151, 0.29995, 0.37439]
# VC_97_mean_n = 0.14800285714285713
# VC_99_n = [0.16794, 0.3806, 0.37099, 0.17438, 0.37099, 0.17438, 0.15744]
# VC_99_mean_n = 0.2566742857142857

CCC_97 = [0.08715, 0.00917, 0.0, 0.00388, 0, 0.02699, 0.00464]
CCC_97_mean = 0.018832857142857144
CCC_99 = [0.0132, 0.00028, 0, 0.0, 0, 0.0009, 0.0]
CCC_99_mean = 0.0020542857142857142

fhs_97 = [0.04281, 0.40809, 0.33008, 3e-05, 0.13337, 0.29673, 0]
fhs_97_mean = 0.17301571428571427
fhs_99 = [0.00673, 0.28774, 0.00012, 0, 0.00637, 0.1023, 0]
fhs_99_mean = 0.05760857142857143

fig, ax = plt.subplots(1, 2)

ax[0].plot(hs_97, color="red", label="HS")
ax[0].plot(VC_97_t3, color="blue", label="VC t_3")
# ax[0].plot(VC_97_n, color="orange", label="VC n")
ax[0].plot(CCC_97, color="green", label="CCC")
ax[0].plot(fhs_97, color="purple", label="FHS")
ax[0].set_ylim(0, 0.5)
ax[0].set_title("P-values CVaR", fontsize=24)
ax[0].set_xlabel("Years", fontsize=24)
ax[0].set_ylabel("P-value", fontsize=24)
ax[0].hlines(0.05, 0, 6.2, label="5% confidence", linestyle="--")
ax[0].legend(fontsize=20)

ax[1].hlines(hs_97_mean, 0, 7, color="red", label="HS mean")
ax[1].hlines(VC_97_mean_t3, 0, 7, color="blue", label="VC t_3 mean")
# ax[1].hlines(VC_97_mean_n, 0, 7, color="orange", label="VC n mean")
ax[1].hlines(CCC_97_mean, 0, 7, color="green", label="CCC mean")
ax[1].hlines(fhs_97_mean, 0, 7, color="purple", label="FHS mean")
ax[1].hlines(0.05, 0, 7, label="5% confidence", linestyle="--")
ax[1].set_ylim(0, 0.5)
ax[1].legend(fontsize=20)
ax[1].set_title("P-values CVaR mean", fontsize=24)
ax[1].set_xlabel("Years", fontsize=24)
ax[1].set_ylabel(" ")
plt.show()
