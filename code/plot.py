import matplotlib.pyplot as plt
import numpy as np

hs_97 = [0.01624, 0.09151, 0.1767, 0.00343, 0.09151, 0.29995, 0.16833]
hs_97_mean = 0.12109571428571428
hs_99 = [0.16794, 0.3806, 0.17438, 0.0, 0.3806, 0.05819, 0.00128]
hs_99_mean = 0.16614142857142858

VC_97_t3 = [0.1767, 0.29995, 0.09151, 0.00343, 0.29995, 0.45183, 0.16833]
VC_97_mean_t3 = 0.2131
VC_99_t3 = [0.16794, 0.37099, 0.37099, 0.05531, 0.05531, 0.17438, 0.5]
VC_99_mean_t3 = 0.24213142857142858

VC_97_n = [0.04137, 0.1767, 0.04137, 0.01073, 0.09151, 0.29995, 0.37439]
VC_97_mean_n = 0.14800285714285713
VC_99_n = [0.16794, 0.3806, 0.37099, 0.17438, 0.37099, 0.17438, 0.15744]
VC_99_mean_n = 0.2566742857142857

CCC_97 = [0.45183, 0.29995, 0.09151, 0.13799, 0.09151, 0.09151, 0.5]
CCC_97_mean = 0.23775714285714283
CCC_99 = [0.3806, 0.3806, 0.37099, 0.16794, 0.16794, 0.37099, 0.5]
CCC_99_mean = 0.33415142857142854

fhs_97 = [0.29995, 0.45183, 0.09151, 0.02895, 0.3888, 0.09151, 0.16833]
fhs_97_mean = 0.21726857142857142
fhs_99 = [0.37099, 0.37099, 0.37099, 0.05531, 0.37099, 0.37099, 0.15744]
fhs_99_mean = 0.29538571428571425

fig, ax = plt.subplots(1, 2)

ax[0].plot(hs_97, color="red", label="HS")
ax[0].plot(VC_97_t3, color="blue", label="VC t_3")
ax[0].plot(VC_97_n, color="orange", label="VC n")
ax[0].plot(CCC_97, color="green", label="CCC")
ax[0].plot(fhs_97, color="purple", label="FHS")
ax[0].set_ylim(0, 0.5)
ax[0].set_title("P-values VaR methods", fontsize=24)
ax[0].set_xlabel("Years", fontsize=24)
ax[0].set_ylabel("P-value", fontsize=24)
ax[0].hlines(0.05, 0, 6.2, label="5% confidence", linestyle="--")
ax[0].legend(fontsize=20)

ax[1].hlines(hs_97_mean, 0, 7, color="red", label="HS mean")
ax[1].hlines(VC_97_mean_t3, 0, 7, color="blue", label="VC t_3 mean")
ax[1].hlines(VC_97_mean_n, 0, 7, color="orange", label="VC n mean")
ax[1].hlines(CCC_97_mean, 0, 7, color="green", label="CCC mean")
ax[1].hlines(fhs_97_mean, 0, 7, color="purple", label="FHS mean")
ax[1].hlines(0.05, 0, 7, label="5% confidence", linestyle="--")
ax[1].set_ylim(0, 0.5)
ax[1].legend(fontsize=20)
ax[1].set_title("P-values VaR methods mean", fontsize=24)
ax[1].set_xlabel("Years", fontsize=24)
ax[1].set_ylabel(" ")
plt.show()
