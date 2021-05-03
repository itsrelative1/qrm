import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


def read_data(lower, upper, weights, removeStress=False):
    assets = pd.read_csv("combined2.csv", sep=",", index_col=0)
    assets["STOXX"] = assets["STOXX"] * assets["FX"]
    assets.drop("FX", axis=1, inplace=True)
    assets = -assets
    asset_returns = assets.pct_change()
    asset_returns = asset_returns.dropna()
    asset_returns.index = pd.to_datetime(asset_returns.index)
    asset_returns = asset_returns[lower:upper]

    if removeStress == True:
        asset_returns["portfolio"] = asset_returns.dot(weights)
        asset_returns = asset_returns[abs(asset_returns["portfolio"]) < 0.05]
        asset_returns.drop("portfolio", axis=1, inplace=True)

        means, cov_matrix = getMetric(asset_returns, lower, upper)

    else:
        means, cov_matrix = getMetric(asset_returns, lower, upper)

    return asset_returns, means, cov_matrix


def read_data_2(lower, upper, weights, h=5):

    assets = pd.read_csv("combined2.csv", sep=",", index_col=0)
    assets["STOXX"] = assets["STOXX"] * assets["FX"]
    assets.drop("FX", axis=1, inplace=True)

    if h == 5:
        count = 0

        for index, row in assets.iterrows():
            count += 1

            if count % 5 != 0:
                assets.drop(index, inplace=True)

    elif h == 10:
        count = 0

        for index, row in assets.iterrows():
            count += 1

            if count % 10 != 0:
                assets.drop(index, inplace=True)

    assets = -assets
    asset_returns = assets.pct_change()
    asset_returns = asset_returns.dropna()
    asset_returns.index = pd.to_datetime(asset_returns.index)
    asset_returns = asset_returns[lower:upper]

    means, cov_matrix = getMetric(asset_returns, lower, upper)
    return asset_returns, means, cov_matrix


def getMetric(assetReturns, lower, upper):

    assetReturns = assetReturns[lower:upper]
    cov_matrix = assetReturns.cov()
    mean = assetReturns.mean()

    return mean, cov_matrix


def portfolioPerformance(weights, meanReturns, covMatrix, time):
    portfolioReturn = np.sum(meanReturns * weights) * time
    volatility = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(time)
    return portfolioReturn, volatility


def historicalVar(returns, alpha=95):

    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha, interpolation="higher")

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVar, alpha=5)


def historicalCVar(returns, alpha=95):

    if isinstance(returns, pd.Series):
        belowVar = returns >= historicalVar(returns, alpha=alpha)
        return returns[belowVar].mean()

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVar, alpha=95)


def p_test_VaR(count, expected, N, alpha=95):
    Z = (count - N * (1 - alpha / 100)) / np.sqrt(N * (1 - alpha / 100) * alpha / 100)
    p_1 = round(norm.sf(abs(Z)), 5)  # one-sided
    p_2 = round(norm.sf(abs(Z)) * 2, 5)  # twosided

    return p_1, p_2


def p_test_VaR2(count, expected, sigma, alpha=95):

    Z = (count) / (sigma * np.sqrt(expected))
    p_1 = round(norm.sf(abs(Z)), 5)  # one-sided
    p_2 = round(norm.sf(abs(Z)) * 2, 5)  # twosided

    return p_1, p_2


def p_plot(loss, VaR, alpha):

    years = math.floor(len(loss) / 252)
    rest = len(loss) % 252

    loss_year = [loss[0 + i * 252 : 252 + i * 252] for i in range(years)]
    loss_year.append(loss[252 * years : 252 * years + rest])

    VaR_year = [VaR[0 + i * 252 : 252 + i * 252] for i in range(years)]
    VaR_year.append(VaR[252 * years : 252 * years + rest])

    p_values = []

    for year_loss, year_VaR in zip(loss_year, VaR_year):

        count = 0

        for x, y in zip(year_loss, year_VaR):

            if x > y:
                count += 1

        p1, p2 = p_test_VaR(count, 0, len(year_loss), alpha=alpha)
        p_values.append(p1)
    mean = np.array(p_values).mean()

    return p_values, mean


def stress(loss, VaR):

    years = math.floor(len(loss) / 252)
    rest = len(loss) % 252

    loss_year = [loss[0 + i * 252 : 252 + i * 252] for i in range(years)]
    loss_year.append(loss[252 * years : 252 * years + rest])

    VaR_year = [VaR[0 + i * 252 : 252 + i * 252] for i in range(years)]
    VaR_year.append(VaR[252 * years : 252 * years + rest])

    count_list = []

    for year_loss, year_VaR in zip(loss_year, VaR_year):

        count = 0

        for x, y in zip(year_loss, year_VaR):

            if x > y:
                count += 1

        count_list.append(count)

    return count_list


def p_plot_es(loss, VaR, CVaR, alpha):

    years = math.floor(len(loss) / 252)
    rest = len(loss) % 252

    loss_year = [loss[0 + i * 252 : 252 + i * 252] for i in range(years)]
    loss_year.append(loss[252 * years : 252 * years + rest])

    VaR_year = [VaR[0 + i * 252 : 252 + i * 252] for i in range(years)]
    VaR_year.append(VaR[252 * years : 252 * years + rest])

    CVaR_year = [CVaR[0 + i * 252 : 252 + i * 252] for i in range(years)]
    CVaR_year.append(CVaR[252 * years : 252 * years + rest])

    p_values = []

    for year_loss, data in zip(loss_year, zip(VaR_year, CVaR_year)):

        loss_list = []

        for i, j in zip(year_loss, zip(data[0], data[1])):

            if i > j[0]:

                loss_list.append(i)

        if len(loss_list) > 1:
            average = np.array(loss_list).mean()
            sigma = np.sqrt(np.array(loss_list).var())
            expected = len(loss_list)

            p1, p2 = p_test_VaR2(average, expected, sigma, alpha=97.5)
            p_values.append(p1)

        else:
            p_values.append(0)

    mean = np.array(p_values).mean()

    return p_values, mean


if __name__ == "__main__":
    Time = 1
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    investment = 1000000

    returns, means, cov_matrix = read_data("2013-02-10", "2021-04-01", weights)
    returns["portfolio"] = investment * returns.dot(weights)
    X_t = returns["portfolio"].values

    VaR97 = []
    CVaR97 = []

    VaR99 = []
    CVaR99 = []

    # count25 = 0
    # count1 = 0
    # expected25 = round(len(X_t[100:]) * (0.025), 2)
    # expected1 = round(len(X_t[100:]) * (0.01), 2)

    borders = [
        ("2013-02-10", "2014-02-10"),
        ("2014-03-10", "2016-03-10"),
        ("2016-04-10", "2021-04-01"),
    ]

    # for sample in borders:
    #     lower = sample[0]
    #     upper = sample[1]

    #     returns, means, cov_matrix = read_data(
    #         lower, upper, weights, removeStress=False
    #     )
    #     returns["portfolio"] = investment * returns.dot(weights)
    #     X_t = pd.Series(returns["portfolio"].values)

    #     VaR97.append(historicalVar(X_t, alpha=97.5))
    #     CVaR97.append(historicalCVar(X_t, alpha=97.5))
    #     VaR99.append(historicalVar(X_t, alpha=99))
    #     CVaR99.append(historicalCVar(X_t, alpha=99))

    # print("VaR 97.5", VaR97)
    # print("CVaR 97.5", CVaR97)

    # print("VaR 99", VaR99)
    # print("CVaR 99", CVaR99)

    for i in range(len(X_t) - 100):
        x_temp = pd.Series(X_t[: 100 + i])

        VaR97.append(historicalVar(x_temp, alpha=97.5))
        CVaR97.append(historicalCVar(x_temp, alpha=97.5))

        VaR99.append(historicalVar(x_temp, alpha=99))
        CVaR99.append(historicalCVar(x_temp, alpha=99))

    # plt.plot(X_t[100:])
    # plt.plot(VaR97, label="97.5% VaR")
    # plt.plot(CVaR97, label="97.5% CVaR")

    # for i, x in enumerate(zip(X_t[100:], VaR97)):

    #     if x[0] > x[1]:
    #         count25 += 1
    #         plt.plot(i, x[0], marker="X", color="orange")
    #         x_25, y_25 = i, x[0]

    # x_last = len(X_t[100:]) - 1
    # p25_1, p25_2 = p_test_VaR(count25, 1, len(X_t[100:]), alpha=97.5)

    # plt.plot(
    #     x_25,
    #     y_25,
    #     marker="X",
    #     color="orange",
    #     label=f"Exceeding 97.5% VaR: count = {count25}, expected = {expected25}, p={p25_1}",
    #     linestyle="None",
    # )
    # plt.plot(
    #     x_last,
    #     VaR97[-1],
    #     marker="D",
    #     color="red",
    #     label=f"97.5% VaR T+1 = {round(VaR97[-1],2)}",
    #     linestyle="None",
    # )
    # plt.plot(
    #     x_last,
    #     CVaR97[-1],
    #     marker="D",
    #     color="blue",
    #     label=f"97.5% CVaR T+1 = {round(CVaR97[-1],2)}",
    #     linestyle="None",
    # )
    # plt.title("Historical Simulation: 1-day horizon, investment = 10^6", fontsize=24)
    # plt.xlabel("Data points", fontsize=24)
    # plt.ylabel("Loss", fontsize=24)
    # plt.legend(fontsize=20)
    # plt.show()

    # plt.plot(X_t[100:])
    # plt.plot(VaR99, label="99% VaR")
    # plt.plot(CVaR99, label="99% CVaR")

    # for i, x in enumerate(zip(X_t[100:], VaR99)):

    #     if x[0] > x[1]:
    #         count1 += 1
    #         plt.plot(i, x[0], marker="X", color="orange")
    #         x_1, y_1 = i, x[0]

    # x_last = len(X_t[100:]) - 1
    # p1_1, p1_2 = p_test_VaR(count1, 1, len(X_t[100:]), alpha=99)

    # plt.plot(
    #     x_1,
    #     y_1,
    #     marker="X",
    #     color="orange",
    #     label=f"Exceeding 99% VaR: count = {count1}, expected = {expected1}, p={p1_1}",
    #     linestyle="None",
    # )
    # plt.plot(
    #     x_last,
    #     VaR99[-1],
    #     marker="D",
    #     color="red",
    #     label=f"99% VaR T+1 = {round(VaR99[-1],2)}",
    #     linestyle="None",
    # )
    # plt.plot(
    #     x_last,
    #     CVaR99[-1],
    #     marker="D",
    #     color="blue",
    #     label=f"99% CVaR T+1 = {round(CVaR99[-1],2)}",
    #     linestyle="None",
    # )
    # plt.title("Historical Simulation: 1-day horizon, investment = 10^6", fontsize=24)
    # plt.xlabel("Data points", fontsize=24)
    # plt.ylabel("Loss", fontsize=24)
    # plt.legend(fontsize=20)
    # plt.show()

    # p97_1, p97_1_mean = p_plot(X_t[100:], VaR97, alpha=97.5)
    # p99_1, p99_1_mean = p_plot(X_t[100:], VaR97, alpha=99)

    # print("97.5: ", p97_1, p97_1_mean)
    # print("99: ", p99_1, p99_1_mean)
    # plt.hlines(0.05, 0, 7, colors="k", linestyles="dashed", label="5% confidence")
    # plt.hlines(p97_1_mean, 0, 7, colors="red", linestyles="dashed", label="97.5% mean")
    # plt.hlines(p99_1_mean, 0, 7, colors="blue", linestyles="dashed", label="99% mean")
    # plt.plot(p97_1, label="97.5% VaR", color="red")
    # plt.plot(p99_1, label="99% VaR", color="blue")
    # plt.legend()
    # plt.show()

    pEs_97, pEs97_mean = p_plot_es(X_t[100:], VaR97, CVaR97, alpha=97.5)
    print("97:", pEs_97, pEs97_mean)

    pEs_99, pEs99_mean = p_plot_es(X_t[100:], VaR99, CVaR99, alpha=99)
    print("99", pEs_99, pEs99_mean)

    # H=5 ------------------------------------------------------------------------------------

    # for sample in borders:
    #     lower = sample[0]
    #     upper = sample[1]

    #     returns, means, cov_matrix = read_data_2(lower, upper, weights, h=10)

    #     returns["portfolio"] = investment * returns.dot(weights)
    #     X_t = pd.Series(returns["portfolio"].values)

    #     VaR97.append(historicalVar(X_t, alpha=97.5))
    #     CVaR97.append(historicalCVar(X_t, alpha=97.5))
    #     VaR99.append(historicalVar(X_t, alpha=99))
    #     CVaR99.append(historicalCVar(X_t, alpha=99))

    # print("VaR 97.5", VaR97)
    # print("CVaR 97.5", CVaR97)

    # print("VaR 99", VaR99)
    # print("CVaR 99", CVaR99)

