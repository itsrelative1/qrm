from historic import read_data, portfolioPerformance, p_plot, p_test_VaR, p_plot_es
import numpy as np
import pandas as pd
from scipy.stats import norm, t, multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats


def varParametric(
    portfolioReturn, portfolioStd, distribution="normal", alpha=95, dof=6
):
    """Calculate portfolio VaR given distribution with known parameters"""

    if distribution == "normal":
        VaR = norm.ppf(1 - alpha / 100) * portfolioStd + portfolioReturn

    elif distribution == "studentT":
        VaR = (
            np.sqrt(1 / dof * (dof - 2)) * t.ppf(1 - alpha / 100, dof) * portfolioStd
            + portfolioReturn
        )

    return VaR


def cvarParametric(
    portfolioReturn, portfolioStd, distribution="normal", alpha=95, dof=6
):
    """Calculate portfolio VaR given distribution with known parameters"""

    if distribution == "normal":
        CVaR = (
            1 / (alpha / 100) * norm.pdf(norm.ppf(alpha / 100)) * portfolioStd
            + portfolioReturn
        )

    elif distribution == "studentT":
        x_anu = t.ppf(alpha / 100, dof)
        CVaR = (
            -1
            / (alpha / 100)
            * 1
            / (1 - dof)
            * (dof - 2 + x_anu ** 2)
            * t.pdf(x_anu, dof)
            * portfolioStd
            + portfolioReturn
        )

    return CVaR


if __name__ == "__main__":
    Time = 1
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    investment = 10 ** 6

    varlistNormal_25 = []
    cvarlistNormal_25 = []

    varlistStudent3_25 = []
    cvarlistStudent3_25 = []
    varlistStudent4_25 = []
    cvarlistStudent4_25 = []
    varlistStudent5_25 = []
    cvarlistStudent5_25 = []
    varlistStudent6_25 = []
    cvarlistStudent6_25 = []

    varlistNormal_1 = []
    cvarlistNormal_1 = []

    varlistStudent3_1 = []
    cvarlistStudent3_1 = []
    varlistStudent4_1 = []
    cvarlistStudent4_1 = []
    varlistStudent5_1 = []
    cvarlistStudent5_1 = []
    varlistStudent6_1 = []
    cvarlistStudent6_1 = []

    returns, means, cov_matrix = read_data(
        "2013-02-10", "2021-04-01", weights, removeStress=False
    )
    returns["portfolio"] = investment * returns.dot(weights)
    X_t = returns["portfolio"].values

    for i in range(len(X_t) - 100):

        x_temp = X_t[: 100 + i]
        mean = x_temp.mean()
        sigma = np.sqrt(x_temp.var())

        varlistStudent3_1.append(
            varParametric(mean, sigma, distribution="studentT", alpha=1, dof=3)
        )
        varlistStudent3_25.append(
            varParametric(mean, sigma, distribution="studentT", alpha=2.5, dof=3)
        )

        varlistNormal_1.append(varParametric(mean, sigma, alpha=1))
        varlistNormal_25.append(varParametric(mean, sigma, alpha=2.5))

        cvarlistStudent3_1.append(
            cvarParametric(mean, sigma, distribution="studentT", alpha=1, dof=3)
        )
        cvarlistStudent3_25.append(
            cvarParametric(mean, sigma, distribution="studentT", alpha=2.5, dof=3)
        )

        varlistStudent4_1.append(
            varParametric(mean, sigma, distribution="studentT", alpha=1, dof=4)
        )
        cvarlistStudent4_1.append(
            cvarParametric(mean, sigma, distribution="studentT", alpha=1, dof=4)
        )
        varlistStudent5_1.append(
            varParametric(mean, sigma, distribution="studentT", alpha=1, dof=5)
        )
        cvarlistStudent5_1.append(
            cvarParametric(mean, sigma, distribution="studentT", alpha=1, dof=5)
        )
        varlistStudent6_1.append(
            varParametric(mean, sigma, distribution="studentT", alpha=1, dof=6)
        )
        cvarlistStudent6_1.append(
            cvarParametric(mean, sigma, distribution="studentT", alpha=1, dof=6)
        )

    # p99_1, p99_1_mean = p_plot(X_t[100:], varlistStudent3_1, alpha=99)
    # p97_1, p97_1_mean = p_plot(X_t[100:], varlistStudent3_25, alpha=97.5)

    # p99_1_n, p99_1_n_mean = p_plot(X_t[100:], varlistNormal_1, alpha=99)
    # p97_1_n, p97_1_n_mean = p_plot(X_t[100:], varlistNormal_25, alpha=97.5)

    # print("97.5 n: ", p97_1_n, p97_1_n_mean)
    # print("99 n: ", p99_1_n, p99_1_n_mean)

    # plt.title("P-values Variance-Covariance method", fontsize=24)
    # plt.xlabel("Years", fontsize=24)
    # plt.ylabel("P-value", fontsize=24)
    # plt.hlines(0.05, 0, 7, linestyles="dashed", label="5% significance")
    # plt.plot(p1, label="99% VaR student-t, df=3")
    # plt.plot(p25, label="97.5% VaR student-t, df=3")
    # plt.legend(fontsize=18)
    # plt.show()

    # plt.title("P-values Variance-Covariance method", fontsize=24)
    # plt.xlabel("Years", fontsize=24)
    # plt.ylabel("P-value", fontsize=24)
    # plt.hlines(0.05, 0, 7, linestyles="dashed", label="5% significance")
    # plt.plot(p1_n, label="99% VaR normal")
    # plt.plot(p25_n, label="97.5% VaR normal")
    # plt.legend(fontsize=18)
    # plt.show()

    count1_t3 = 0
    count25_t3 = 0
    expected1 = len(X_t[100:]) * 0.01
    expected25 = round(len(X_t[100:]) * (0.025), 3)

    plt.plot(X_t[100:])
    plt.plot(varlistStudent3_1, label="99% VaR t_3")
    plt.plot(cvarlistStudent3_1, label="99% CVaR t_3")
    x_last = len(X_t[100:]) - 1

    for i, x in enumerate(zip(X_t[100:], varlistStudent3_1)):

        if x[0] > x[1]:
            count1_t3 += 1
            plt.plot(i, x[0], marker="X", color="orange")
            x_1, y_1 = i, x[0]

    p1_t3, p2_t3 = p_test_VaR(count1_t3, 1, len(X_t[100:]), alpha=99)

    plt.plot(
        x_1,
        y_1,
        marker="X",
        color="orange",
        label=f"Exceeding 99% VaR: count = {count1_t3}, expected = {expected1}, p={p1_t3}",
        linestyle="None",
    )
    plt.plot(
        x_last,
        varlistStudent3_1[-1],
        marker="D",
        color="red",
        label=f"99% VaR T+1 = {round(varlistStudent3_1[-1],2)}",
        linestyle="None",
    )
    plt.plot(
        x_last,
        cvarlistStudent3_1[-1],
        marker="D",
        color="blue",
        label=f"99% CVaR T+1 = {round(cvarlistStudent3_1[-1],2)}",
        linestyle="None",
    )

    plt.title("Variance-Covariance: 1-day horizon, investment = 10^6", fontsize=24)
    plt.xlabel("Data points", fontsize=24)
    plt.ylabel("Loss", fontsize=24)
    plt.legend(fontsize=20)
    plt.show()

    plt.plot(X_t[100:])
    plt.plot(varlistStudent3_25, label="97.5% VaR t_3")
    plt.plot(cvarlistStudent3_25, label="97.5% CVaR t_3")
    x_last = len(X_t[100:]) - 1

    for i, x in enumerate(zip(X_t[100:], varlistStudent3_25)):

        if x[0] > x[1]:
            count25_t3 += 1
            plt.plot(i, x[0], marker="X", color="orange")
            x_25, y_25 = i, x[0]

    p25_t3, p252_t3 = p_test_VaR(count25_t3, 1, len(X_t[100:]), alpha=97.5)

    plt.plot(
        x_25,
        y_25,
        marker="X",
        color="orange",
        label=f"Exceeding 97.5% VaR: count = {count25_t3}, expected = {expected25}, p={p25_t3}",
        linestyle="None",
    )
    plt.plot(
        x_last,
        varlistStudent3_25[-1],
        marker="D",
        color="red",
        label=f"97.5% VaR T+1 = {round(varlistStudent3_25[-1],2)}",
        linestyle="None",
    )
    plt.plot(
        x_last,
        cvarlistStudent3_25[-1],
        marker="D",
        color="blue",
        label=f"97.5% CVaR T+1 = {round(cvarlistStudent3_25[-1],2)}",
        linestyle="None",
    )

    plt.title("Variance-Covariance: 1-day horizon, investment = 10^6", fontsize=24)
    plt.xlabel("Data points", fontsize=24)
    plt.ylabel("Loss", fontsize=24)
    plt.legend(fontsize=20)
    plt.show()

    pEs_97, pEs97_mean = p_plot_es(
        X_t[100:], varlistStudent3_25, cvarlistStudent3_25, alpha=97.5
    )
    print("97", pEs_97, pEs97_mean)

    pEs_99, pEs99_mean = p_plot_es(
        X_t[100:], varlistStudent3_1, cvarlistStudent3_1, alpha=99
    )
    print("99", pEs_99, pEs99_mean)

