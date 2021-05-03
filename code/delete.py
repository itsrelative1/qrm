import pandas as pd
import numpy as np

# df1 = pd.read_csv("stoxx.csv", sep=",", parse_dates=True)
# df2 = pd.read_csv("btc.csv", sep=",", parse_dates=True)
# df3 = pd.read_csv("libor.csv", sep=",", parse_dates=True)
# df4 = pd.read_csv("sp500.csv", sep=",", parse_dates=True)

# # newindex = df1.index.union(df2.index)

# # df1 = df1.reindex(newindex)
# # df2 = df2.reindex(newindex)

# df1 = df1.merge(df2, how="left", left_on="Date", right_on="Date")
# df1 = df1.merge(df3, how="left", left_on="Date", right_on="Date")
# df1 = df1.merge(df4, how="left", left_on="Date", right_on="Date")

# df1 = df1.dropna()

# df1.to_csv("combined.csv", index=False)
# df = pd.read_csv("combined.csv", sep=",")
# df = df.dropna()
# df2 = pd.read_csv("FX.csv", sep=",")

# df = df.merge(df2, how="left", left_on="Date", right_on="Date")
# df = df.dropna()
# print(df)

# df.to_csv("combined2.csv", index=False)
