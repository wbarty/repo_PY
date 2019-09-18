from pylab import mpl, plt
import scipy.stats as sm
import pandas as pd
import math
import numpy as np
import statsmodels.api as sm
import numpy as np


def gen_paths(S0, r, sigma, T, M, I):
    '''Generate Monte Carlo paths for geometric Brownian motion.
    Parameters
    ==========
    S0: float
        index stock/index value
    r: float
            constant short rate
     sigma: float
         constant volatility
     T: float
         final time horizon
     M: int
         numbrer of time steps
     I: int
         number of paths to be simulated
     M: int
         number of time steps

     Returns
     =======
     paths: ndarray, shape (M+1, I)
             simulated paths given the parameters
     '''
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rand)
    return paths

#Alternate Variable Prices
S0 = 100.
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000
np.random.seed(1000)
paths = gen_paths(S0, r, sigma, T, M, I)
S0 * math.exp(r * T)

paths[-1].mean()

plt.figure(figsize=(10, 6))

plt.plot(paths[:, :10])

plt.xlabel('time steps')

plt.ylabel('index label')

plt.show()
