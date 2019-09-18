# SABR Volatility Model
# https://en.wikipedia.org/wiki/SABR_volatility_model

import numpy as np
import pandas as pd
import pandas_montecarlo
import math
import scipy.stats
import matplotlib.pyplot as mpl

# Asymptotic Solution - not necessary arbitrage free
# European Call

F =  # underlying forward (rate, swapr rate, stock price)
K =  # strike price
T =  # time to maturity
v =  # forward rate volatility
v0 = v[0]  # forward rate at time 0
p =  # instantaneous correlation between the underlying asset and its volatility

# Calibration Parameters

B =  # beta - a constant, can be calibrated to a variable (controls skewness): 0, 1
a =  # alpha - a constant volatility-like parameter for the forward, controls the height of the ATM implied volatility.
V =  # volvol - the lognormal volatility of the volatility parameter alpha
F0 = F[0]  # F at time 0
Fmid = np.sqrt(F0 * K)  # geometric average between F0 and K
CF = F**B
CFmid = Fmid**B
zeta = (a / v0 * (1 - B)) * (F0 * np.exp(1 - B) - K * np.exp(1 - B))
gamma1 = B / Fmid
gamma2 = -B * (1 - B) / Fmid**2
Dzeta = np.log((np.sqrt(1 - 2 * p * zeta + zeta**2) + zeta - p) / (1 - p))
epsilon = T * (a**2)  # the approximate solution error from asymptopic expansion

# Lognormal/normal Implied Volatility
# normal is usually more accurate for regualr market data


def impvol(a, F0, K, Dzeta, gamma1, gamma2, Fmid, v0, CFmid, p, epsilon, option='lognormal'):
    if option == 'lognormal':
        logimpvol = a(np.log(F0 / K) / Dzeta) * (1 + ((((2 * gamma2 - gamma1**2 + 1) / (Fmid**2) / 24) * ((v0 * CFmid) / a)**2 + ((p * gamm1) / 4) * ((v0 * CFmid) / a) + (2 - 3(p)**2) / 24) * epsilon))

    if option == 'normal':
        normimpvol = a((F0 - K) / Dzeta) * (1 + ((((2 * gamma2 - gamma1**2 + 1) / 24) * ((v0 * CFmid) / a)**2 + ((p * gamm1) / 4) * ((v0 * CFmid) / a) + (2 - 3(p)**2) / 24) * epsilon))

    return impvol

# mcimpvol = impvol.montecarlo(
#     sims=10,
#     bust=-0.1,  # probability of going bust
#     goal=0.9  # probaility of reaching 100% goal
# )
# mc.plot(mcimpvol, title='Implied Volatility MCS')
