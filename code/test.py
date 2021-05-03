import numpy as np
from scipy.stats import norm
import math
import pandas as pd
from historic import historicalCVar, historicalVar

a = [5, 2, 3, 21, 50, 6, 31, 8, 4, 10]
b = [1, 1, 1, 1, 50, 6, 31, 8, 4, 10]
c = [6, 6, 6, 6, 50, 6, 6, 6, 6, 6]


for i, j in zip(a, zip(b, c)):
    print(i, j[0], j[1])

