from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from historic import (
    portfolioPerformance,
    historicalVar,
    historicalCVar,
    p_test_VaR,
    p_plot,
    p_plot_es,
)


def read_data(lower, upper, weights, removeStress=False):
    assets = pd.read_csv("combined2.csv", sep=",", index_col=0)
    assets["STOXX"] = assets["STOXX"] * assets["FX"]
    assets.drop("FX", axis=1, inplace=True)

    asset_returns = assets.pct_change()
    asset_returns = -asset_returns
    asset_returns = asset_returns.dropna()
    asset_returns.index = pd.to_datetime(asset_returns.index)

    if removeStress == True:
        asset_returns["portfolio"] = asset_returns.dot(weights)
        asset_returns = asset_returns[abs(asset_returns["portfolio"]) < 0.1]
        asset_returns.drop("portfolio", axis=1, inplace=True)

    return asset_returns


weights = np.array([0.25, 0.25, 0.25, 0.25])
investment = 10 ** 6
sigma_steps = 100
l = 0.98

# Looking at the Closing Price in USD
returns = read_data("2013-02-10", "2021-04-01", weights)
returns["portfolio"] = returns.dot(weights) * investment

# 1) Sigma_0 van portfolio
X_t = returns["portfolio"].values
plt.plot(X_t)
plt.show()
# sigma_0 = np.sqrt(X_t[: sigma_steps + 1].var())
# X_full = X_t
# X_t = X_t[100:]

# # 2) EWMA serie
# s_hat = np.zeros(len(X_t) + 1)
# s_hat[0] = sigma_0

# for i in range(len(s_hat) - 1):
#     s_hat[i + 1] = np.sqrt(l * s_hat[i] ** 2 + (1 - l) * X_t[i] ** 2)


# # 3) Z lijst die distributie voorsteld door returns te delen door volatility
# z_hat = np.zeros(len(X_t))

# for i in range(len(X_t)):
#     z_hat[i] = X_t[i] / s_hat[i]

# VaR_1_list = []
# VaR_25_list = []
# CVaR_1_list = []
# CVaR_25_list = []

# for i in range(len(z_hat)):
#     z_temp = z_hat[: 100 + i]
#     x_temp = [j * s_hat[i + 1] for j in z_temp]
#     x_temp = pd.Series(x_temp)

#     # Calculate VaR CVaR for 1-day horizon
#     VaR_1_list.append(historicalVar(x_temp, alpha=99))
#     VaR_25_list.append(historicalVar(x_temp, alpha=97.5))
#     CVaR_1_list.append(historicalCVar(x_temp, alpha=99))
#     CVaR_25_list.append(historicalCVar(x_temp, alpha=97.5))


# x = np.arange(1612)
# x_last = x[-1]
# plt.plot(x, X_t)
# plt.plot(VaR_25_list, label="97.5% VaR")
# plt.plot(CVaR_25_list, label="97.5% CVaR")

# count25 = 0
# count1 = 0
# expected25 = round(len(returns) * (0.025), 2)
# expected1 = round(len(returns) * (0.01), 2)

# for i, x in enumerate(zip(X_t, VaR_25_list)):
#     if x[0] > x[1]:
#         count25 += 1
#         plt.plot(i, x[0], marker="X", color="orange")
#         x_25, y_25 = i, x[0]

# # plt.plot(
# #     x_25,
# #     y_25,
# #     marker="X",
# #     color="orange",
# #     label=f"Exceeding 97.5% VaR: count = {count25}, expected = {expected25}, p={p25_1}",
# #     linestyle="None",
# # )
# # plt.plot(
# #     x_last,
# #     VaR_25_list[-1],
# #     marker="D",
# #     color="red",
# #     label=f"97.5% VaR T+1 = {round(VaR_25_list[-1],2)}",
# #     linestyle="None",
# # )
# # plt.plot(
# #     x_last,
# #     CVaR_25_list[-1],
# #     marker="D",
# #     color="blue",
# #     label=f"97.5% CVaR T+1 = {round(CVaR_25_list[-1],2)}",
# #     linestyle="None",
# # )

# # plt.title("FHS-EWMA: 1-day horizon, investment = 10^6", fontsize=24)
# # plt.xlabel("Data points", fontsize=24)
# # plt.ylabel("Loss", fontsize=24)
# # plt.xticks(fontsize=14)
# # plt.yticks(fontsize=14)
# # plt.legend(fontsize=16)
# # plt.show()
# # plt.clf()

# # x = np.arange(1612)
# # x_last = x[-1]
# # plt.plot(x, X_t)
# # plt.plot(VaR_1_list, label="99% VaR")
# # plt.plot(CVaR_1_list, label="99% CVaR")

# for i, x in enumerate(zip(X_t, VaR_1_list)):
#     if x[0] > x[1]:
#         count1 += 1
#         plt.plot(i, x[0], marker="X", color="orange")
#         x_1, y_1 = i, x[0]


# # p99_1, p99_1_mean = p_plot(X_t, VaR_1_list, alpha=99)
# # p97_1, p97_1_mean = p_plot(X_t, VaR_25_list, alpha=97.5)

# # print("99:", p99_1, p99_1_mean)
# # print("97.5:", p97_1, p97_1_mean)

# pEs_97, pEs97_mean = p_plot_es(X_t[100:], VaR_25_list, CVaR_25_list, alpha=97.5)
# print("97", pEs_97, pEs97_mean)

# pEs_99, pEs99_mean = p_plot_es(X_t[100:], VaR_1_list, CVaR_1_list, alpha=99)
# print("99", pEs_99, pEs99_mean)

# # plt.plot(
# #     x_1,
# #     y_1,
# #     marker="X",
# #     color="orange",
# #     label=f"Exceeding 99% VaR: count = {count1}, expected = {expected1}, p={p1_1}",
# #     linestyle="None",
# # )
# # plt.plot(
# #     x_last,
# #     VaR_1_list[-1],
# #     marker="D",
# #     color="red",
# #     label=f"99% VaR T+1 = {round(VaR_1_list[-1],2)}",
# #     linestyle="None",
# # )
# # plt.plot(
# #     x_last,
# #     CVaR_1_list[-1],
# #     marker="D",
# #     color="blue",
# #     label=f"99% CVaR T+1 = {round(CVaR_1_list[-1],2)}",
# #     linestyle="None",
# # )

# # plt.title("FHS-EWMA: 1-day horizon, investment = 10^6", fontsize=24)
# # plt.xlabel("Data points", fontsize=24)
# # plt.ylabel("Loss", fontsize=24)
# # plt.xticks(fontsize=14)
# # plt.yticks(fontsize=14)
# # plt.legend(fontsize=16)
# # plt.show()

