from historic import read_data, portfolioPerformance, p_plot
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
    # Subsamples = np.arange(0.1, 1.05, 0.05)
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

    # # plt.vlines("2020-02-01", colors="red", linestyles="--")
    # # plt.vlines("2020-06-01", colors="red", linestyles="--")
    # plt.ylabel("Portfolio return", fontsize=18)
    # plt.title("Portfolio Return", fontsize=18)
    # plt.xlabel("Date", fontsize=18)
    # plt.show()
    # plt.clf()

    # fig, ax = plt.subplots(1, 5, sharey=True)

    # scipy.stats.probplot(returns["portfolio"], dist="norm", plot=ax[0])
    # scipy.stats.probplot(returns["portfolio"], dist=t(6), plot=ax[1])
    # scipy.stats.probplot(returns["portfolio"], dist=t(5), plot=ax[2])
    # scipy.stats.probplot(returns["portfolio"], dist=t(4), plot=ax[3])
    # scipy.stats.probplot(returns["portfolio"], dist=t(3), plot=ax[4])

    # ax[0].set_title("Normal", fontsize=22)
    # ax[0].tick_params(axis="both", labelsize=14)
    # ax[0].set_ylabel("Return", fontsize=22)
    # ax[1].set_title("student-t, Df = 6", fontsize=22)
    # ax[1].tick_params(axis="both", labelsize=14)
    # ax[1].set_ylabel(" ")
    # ax[2].set_title("student-t, Df = 5", fontsize=22)
    # ax[2].tick_params(axis="both", labelsize=14)
    # ax[2].set_ylabel(" ")
    # ax[3].set_title("student-t, Df = 4", fontsize=22)
    # ax[3].tick_params(axis="both", labelsize=14)
    # ax[3].set_ylabel(" ")
    # ax[4].set_title("student-t, Df = 3", fontsize=22)
    # ax[4].tick_params(axis="both", labelsize=14)
    # ax[4].set_ylabel(" ")

    # plt.show()
    borders = [
        ("2013-02-10", "2014-02-10"),
        ("2014-03-10", "2016-03-10"),
        ("2016-04-10", "2021-04-01"),
    ]

    for sample in borders:
        lower = sample[0]
        upper = sample[1]

        returns, means, cov_matrix = read_data(lower, upper, weights, removeStress=True)

        returns["portfolio"] = investment * returns.dot(weights)
        portfolioReturn, volatility = portfolioPerformance(
            weights, means, cov_matrix, 1
        )

        varlistNormal_1.append(
            investment * varParametric(portfolioReturn, volatility, alpha=1)
        )
        cvarlistNormal_1.append(
            investment * cvarParametric(portfolioReturn, volatility, alpha=1)
        )

        varlistNormal_25.append(
            investment * varParametric(portfolioReturn, volatility, alpha=2.5)
        )
        cvarlistNormal_25.append(
            investment * cvarParametric(portfolioReturn, volatility, alpha=2.5)
        )

        varlistStudent3_1.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=3, alpha=1
            )
        )
        cvarlistStudent3_1.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=3, alpha=1
            )
        )

        varlistStudent4_1.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=4, alpha=1
            )
        )
        cvarlistStudent4_1.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=4, alpha=1
            )
        )

        varlistStudent5_1.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=5, alpha=1
            )
        )
        cvarlistStudent5_1.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=5, alpha=1
            )
        )

        varlistStudent6_1.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=6, alpha=1
            )
        )
        cvarlistStudent6_1.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=6, alpha=1
            )
        )

        varlistStudent3_25.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=3, alpha=2.5
            )
        )
        cvarlistStudent3_25.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=3, alpha=2.5
            )
        )

        varlistStudent4_25.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=4, alpha=2.5
            )
        )
        cvarlistStudent4_25.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=4, alpha=2.5
            )
        )

        varlistStudent5_25.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=5, alpha=2.5
            )
        )
        cvarlistStudent5_25.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=5, alpha=2.5
            )
        )

        varlistStudent6_25.append(
            investment
            * varParametric(
                portfolioReturn, volatility, distribution="studentT", dof=6, alpha=2.5
            )
        )
        cvarlistStudent6_25.append(
            investment
            * cvarParametric(
                portfolioReturn, volatility, distribution="studentT", dof=6, alpha=2.5
            )
        )

    p97_1, p97_1_mean = p_plot(X_t[100:], varlistStudent3_25, alpha=97.5)
    p99_1, p99_1_mean = p_plot(X_t[100:], varlistStudent3_1, alpha=99)

    print("97.5: ", p97_1, p97_1_mean)
    print("99: ", p99_1, p99_1_mean)

    # dates = ["2013-2014", "2014-2016", "2016-2021"]
    # x = np.arange(len(dates))
    # widths = np.array([0.05, 0.05, 0.05])

    # plt.bar(x, varlistStudent3_25, width=0.05, label="t_3")
    # plt.bar(x + widths, varlistStudent4_25, width=0.05, label="t_4")
    # plt.bar(x + 2 * widths, varlistStudent5_25, width=0.05, label="t_5")
    # plt.bar(x + 3 * widths, varlistStudent6_25, width=0.05, label="t_6")
    # plt.bar(x + widths, varlistNormal_25, width=0.05, label="Normal 97.5")
    # print("97.5 var", varlistNormal_25)
    # print("99 var", varlistNormal_1)
    # print("97.5 cvar", cvarlistNormal_25)
    # print("99 cvar", cvarlistNormal_1)
    # plt.bar(x, cvarlistStudent3_25, width=0.05, label="t_3")
    # plt.bar(x + widths, cvarlistStudent4_25, width=0.05, label="t_4")
    # plt.bar(x + 2 * widths, cvarlistStudent5_25, width=0.05, label="t_5")
    # plt.bar(x + 3 * widths, cvarlistStudent6_25, width=0.05, label="t_6")
    # plt.bar(x + 4 * widths, cvarlistNormal_25, width=0.05, label="Normal")

    # plt.bar(x, cvarlistStudent3_1, width=0.05, label="t_3")
    # plt.bar(x + widths, cvarlistStudent4_1, width=0.05, label="t_4")
    # plt.bar(x + 2 * widths, cvarlistStudent5_1, width=0.05, label="t_5")
    # plt.bar(x + 3 * widths, cvarlistStudent6_1, width=0.05, label="t_6")
    # plt.bar(x + 4 * widths, cvarlistNormal_1, width=0.05, label="Normal")

    # plt.bar(x, varlistStudent3_1, width=0.05, label="t_3")
    # plt.bar(x + widths, varlistStudent4_1, width=0.05, label="t_4")
    # plt.bar(x + 2 * widths, varlistStudent5_1, width=0.05, label="t_5")
    # plt.bar(x + 3 * widths, varlistStudent6_1, width=0.05, label="t_6")
    # plt.bar(x, varlistNormal_1, width=0.05, label="Normal 99")

    # plt.plot(dates, varlistNormal_25, marker="P", color="blue", label="Normal VaR")
    # plt.plot(
    #     dates,
    #     cvarlistNormal_25,
    #     marker="X",
    #     linestyle="--",
    #     color="blue",
    #     label="Normal CVaR",
    # )
    # plt.plot(dates, varlistStudent3_25, marker="P", color="red", label="t_3 VaR")
    # plt.plot(
    #     dates,
    #     cvarlistStudent3_25,
    #     marker="X",
    #     linestyle="--",
    #     color="red",
    #     label="t_3 CVaR",
    # )
    # plt.plot(dates, varlistStudent4_25, marker="P", color="green", label="t_4 VaR")
    # plt.plot(
    #     dates,
    #     cvarlistStudent4_25,
    #     marker="X",
    #     color="green",
    #     linestyle="--",
    #     label="t_4 CVaR",
    # )
    # plt.plot(dates, varlistStudent5_25, marker="P", color="purple", label="t_5 VaR")
    # plt.plot(
    #     dates,
    #     cvarlistStudent5_25,
    #     marker="X",
    #     color="purple",
    #     linestyle="--",
    #     label="t_5 CVaR",
    # )
    # plt.plot(dates, varlistStudent6_25, marker="P", color="black", label="t_6 VaR")
    # plt.plot(
    #     dates,
    #     cvarlistStudent6_25,
    #     marker="X",
    #     color="black",
    #     linestyle="--",
    #     label="t_6 CVaR",
    # )
    # plt.xlabel("Samples", fontsize=24)
    # plt.ylabel("Loss", fontsize=24)
    # plt.title("VaR and CVaR with stress periods", fontsize=24)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend()

    # plt.legend(fontsize=18)
    # plt.show()

