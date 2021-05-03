import datetime as dt
import sys
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from historic import read_data, portfolioPerformance, p_test_VaR, p_plot, p_plot_es
from parametric import varParametric, cvarParametric

weights = np.array([0.25, 0.25, 0.25, 0.25])
model = "Parametric"

# Looking at the Closing Price in USD
returns, means, cov_matrix = read_data("2013-02-10", "2021-04-01", weights)

investment = 10 ** 6
corr_matrix = returns.corr()
portfolioReturn = np.sum(means * weights)
VaR_CCC_25 = []
VaR_CCC_1 = []
CVaR_CCC_25 = []
CVaR_CCC_1 = []

std_df = pd.DataFrame()

# Run Garch(1, 1) for std of all assets
for loc, column in enumerate(returns):

    returns = returns.sort_index()
    Garch = arch_model(
        returns[column], vol="Garch", p=1, o=0, q=1, dist="normal", rescale=False
    )
    residual = Garch.fit(disp="off")

    forecasts = residual.forecast(start="2013-02-10", reindex=False,)
    mean = forecasts.mean
    var = forecasts.variance
    std_df[column] = np.sqrt(var["h.1"])


std_df_matrix = std_df.values

print(len(std_df_matrix))

for i in range(len(std_df_matrix)):

    rows = corr_matrix.shape[0]
    std_matrix = np.zeros((rows, rows))

    for j in range(rows):
        std_matrix[j, j] = std_df_matrix[i, j]

    cov_matrix = np.matmul(np.matmul(std_matrix, corr_matrix), std_matrix)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    VaR_CCC_25.append(
        investment
        * varParametric(
            portfolioReturn, volatility, distribution="normal", alpha=2.5, dof=6
        )
    )

    VaR_CCC_1.append(
        investment
        * varParametric(
            portfolioReturn, volatility, distribution="normal", alpha=1, dof=6
        )
    )

    CVaR_CCC_25.append(
        investment
        * cvarParametric(
            portfolioReturn, volatility, distribution="normal", alpha=2.5, dof=6
        )
    )

    CVaR_CCC_1.append(
        investment
        * cvarParametric(
            portfolioReturn, volatility, distribution="normal", alpha=1, dof=6
        )
    )

returns["portfolio"] = investment * returns.dot(weights)
returns = returns.sort_index()
portfolio_returns = returns["portfolio"].values
plt.plot(portfolio_returns)
counts_25 = 0
counts_1 = 0
x_25 = 0
x_1 = 0
y_25 = 0
y_1 = 0
expected1 = round(len(returns) * (0.01), 2)
expected25 = round(len(returns) * (0.025), 2)

# for x, y in enumerate(zip(portfolio_returns, VaR_CCC_1)):

#     if y[0] > y[1]:
#         plt.plot(x, y[0], marker="X", color="k")
#         x_1 = x
#         y_1 = y[0]
#         counts_1 += 1
#     x_last = x

# p1_1, p1_2 = p_test_VaR(counts_1, expected1, len(returns), alpha=99)

# plt.plot(
#     x_1,
#     y_1,
#     marker="X",
#     color="k",
#     label=f"Exceeding 99% VaR: count = {counts_1}, expected = {expected1}, p={p1_1}",
#     linestyle="None",
# )
# plt.plot(
#     x_last,
#     VaR_CCC_1[-1],
#     marker="D",
#     color="red",
#     label=f"99% VaR T+1 = {round(VaR_CCC_1[-1],2)}",
#     linestyle="None",
# )
# plt.plot(
#     x_last,
#     CVaR_CCC_1[-1],
#     marker="D",
#     color="blue",
#     label=f"99% CVaR T+1 = {round(CVaR_CCC_1[-1],2)}",
#     linestyle="None",
# )

# plt.plot(
#     VaR_CCC_1, label=f"99% VaR",
# )
# plt.plot(
#     CVaR_CCC_1, label=f"99% CVaR",
# )

# plt.xlabel("Data points", fontsize=24)
# plt.ylabel("Loss", fontsize=24)
# plt.legend(fontsize=18)
# plt.title("CCC-Garch(1, 1), 1-day horizon, investment = 10^6", fontsize=24)
# plt.yticks(fontsize=14)

# plt.show()

# plt.plot(portfolio_returns)

# for x, y in enumerate(zip(portfolio_returns, VaR_CCC_25)):

#     if y[0] > y[1]:
#         plt.plot(x, y[0], marker="X", color="k")
#         x_25 = x
#         y_25 = y[0]
#         counts_25 += 1

# p25_1, p25_2 = p_test_VaR(counts_25, expected25, len(returns), alpha=97.5)

# plt.plot(
#     x_25,
#     y_25,
#     marker="X",
#     color="k",
#     label=f"Exceeding 97.5% VaR: count = {counts_25}, expected = {round(len(returns)*(0.025), 2)}, p={p25_1}",
#     linestyle="None",
# )

# plt.plot(
#     x_last,
#     VaR_CCC_25[-1],
#     marker="D",
#     color="red",
#     label=f"97.5% VaR T+1 = {round(VaR_CCC_25[-1],2)}",
#     linestyle="None",
# )
# plt.plot(
#     x_last,
#     CVaR_CCC_25[-1],
#     marker="D",
#     color="blue",
#     label=f"97.5% CVaR T+1 = {round(CVaR_CCC_25[-1],2)}",
#     linestyle="None",
# )

# plt.plot(
#     VaR_CCC_25, label=f"97.5% VaR",
# )
# plt.plot(
#     CVaR_CCC_25, label=f"97.5% CVaR",
# )

# plt.xlabel("Data points", fontsize=24)
# plt.ylabel("Loss", fontsize=24)
# plt.legend(fontsize=18)
# plt.title("CCC-Garch(1, 1), 1-day horizon, investment = 10^6", fontsize=24)
# plt.yticks(fontsize=14)

# plt.show()

# p_1, p_1_mean = p_plot(portfolio_returns, VaR_CCC_1, 99)
# p_25, p_25_mean = p_plot(portfolio_returns, VaR_CCC_25, 97.5)

# print("99:", p_1, p_1_mean)
# print("97.5:", p_25, p_25_mean)

pEs_97, pEs97_mean = p_plot_es(portfolio_returns, VaR_CCC_25, CVaR_CCC_25, alpha=97.5)
print("97", pEs_97, pEs97_mean)

pEs_99, pEs99_mean = p_plot_es(portfolio_returns, VaR_CCC_1, CVaR_CCC_1, alpha=99)
print("99", pEs_99, pEs99_mean)

