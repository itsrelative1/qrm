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
    stress,
)


def read_data(lower, upper, weights, scenario):
    assets = pd.read_csv("combined2.csv", sep=",", index_col=0)

    if scenario == "SP20":

        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.SP500 += update.SP500 * 0.2

            else:
                update.SP500 -= update.SP500 * 0.2

            assets.update(update)

    elif scenario == "SP40":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.SP500 += update.SP500 * 0.4

            else:
                update.SP500 -= update.SP500 * 0.4

            assets.update(update)

    elif scenario == "STOXX20":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.STOXX += update.STOXX * 0.2

            else:
                update.STOXX -= update.STOXX * 0.2

            assets.update(update)

    elif scenario == "STOXX40":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.STOXX += update.STOXX * 0.4

            else:
                update.STOXX -= update.STOXX * 0.4

            assets.update(update)

    elif scenario == "BTC20":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.BTC += update.BTC * 0.2

            else:
                update.BTC -= update.BTC * 0.2

            assets.update(update)

    elif scenario == "BTC40":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.BTC += update.BTC * 0.4

            else:
                update.BTC -= update.BTC * 0.4

            assets.update(update)

    elif scenario == "LIB2":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.LIBOR += update.LIBOR * 0.02

            else:
                update.LIBOR -= update.LIBOR * 0.02

            assets.update(update)

    elif scenario == "LIB3":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.LIBOR += update.LIBOR * 0.03

            else:
                update.LIBOR -= update.LIBOR * 0.03

            assets.update(update)

    elif scenario == "FX":
        for _ in range(6):
            update = assets.sample()
            r = np.random.random()

            if r > 0.5:
                update.FX += update.FX * 0.1

            else:
                update.FX -= update.FX * 0.1

            assets.update(update)

    assets["STOXX"] = assets["STOXX"] * assets["FX"]
    assets.drop("FX", axis=1, inplace=True)
    assets = -assets
    asset_returns = assets.pct_change()
    asset_returns = asset_returns.dropna()
    asset_returns.index = pd.to_datetime(asset_returns.index)

    return asset_returns


weights = np.array([0.25, 0.25, 0.25, 0.25])
investment = 10 ** 6
sigma_steps = 100
l = 0.98

# Looking at the Closing Price in USD
returns = read_data("2013-02-10", "2021-04-01", weights, "FX")
returns["portfolio"] = returns.dot(weights) * investment

# 1) Sigma_0 van portfolio
X_t = returns["portfolio"].values
sigma_0 = np.sqrt(X_t[: sigma_steps + 1].var())
X_full = X_t
X_t = X_t[100:]

# 2) EWMA serie
s_hat = np.zeros(len(X_t) + 1)
s_hat[0] = sigma_0

for i in range(len(s_hat) - 1):
    s_hat[i + 1] = np.sqrt(l * s_hat[i] ** 2 + (1 - l) * X_t[i] ** 2)


# 3) Z lijst die distributie voorsteld door returns te delen door volatility
z_hat = np.zeros(len(X_t))

for i in range(len(X_t)):
    z_hat[i] = X_t[i] / s_hat[i]

VaR_1_list = []
VaR_25_list = []
CVaR_1_list = []
CVaR_25_list = []

for i in range(len(z_hat)):
    z_temp = z_hat[: 100 + i]
    x_temp = [j * s_hat[i + 1] for j in z_temp]
    x_temp = pd.Series(x_temp)

    # Calculate VaR CVaR for 1-day horizon
    VaR_1_list.append(historicalVar(x_temp, alpha=99))
    VaR_25_list.append(historicalVar(x_temp, alpha=97.5))
    CVaR_1_list.append(historicalCVar(x_temp, alpha=99))
    CVaR_25_list.append(historicalCVar(x_temp, alpha=97.5))

count_list = stress(X_t[100:], VaR_1_list)
print(count_list)
