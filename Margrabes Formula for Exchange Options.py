# Margrabes Formula option pricing for an exchange of assets on maturity (Exchange Options)

import numpy as np
import pandas as pd
import scipy.stats

# suppose S1 and S2 have constant dividend yields q1 and q2, the option gives the right to exchange S2 for S1 at maturity (EURO) T.
# payoff of C(T) = max(0, S1(T) - S2(T))

# for to calculation of volatility one can use the porfolio standard deviation formula with the correlation coefficient between S1 and S2

# S1 = pd.read_csv('prices') or through API connection retrieve values
# S2 = pd.read_csv('prices') or through API connection retrieve values

prices = pd.DataFrame(S1, S2)  # convert to DataFrame for extraction
p = df.corr(prices, method='pearson')  # correlation of df pearsons corr coefficient
var1 = df.var(prices, axis=0)[0]  # variance of S1
var2 = df.var(prices, axis=0)[1]  # variance of S2
vol = np.sqrt(var1 + var2 - 2 * (np.sqrt(var1)) * (np.sqrt(var2)) * p)  # portfolio SD

S1_0 = S1[0]  # S1 at time 0
S2_0 = S2[0]  # S2 at time 0
q1 =  # expected dividend yield under risk neutrality
q2 =
T =  # TTM
v = vol


def exOP(S1_0, S2_0, q1, q2, T, v, option='call'):
    d1 = (np.log(S1_0 / S2_0)) + (q2 - q1 + v / 2) / np.sqrt(v) * np.sqrt(T)
    d2 = d1 - np.sqrt(v) * np.sqrt(T)
    if option == 'call':
        exOP = np.exp(-q1 * T) * S1_0 * norm.cdf(d1) - np.exp(-q2 * T) * S2_0 * norm.cdf(d2)
    if option == 'put':
        exOP = np.exp(-q2 * T) * S2_0 * norm.cdf(-d2) - np.exp(-q1 * T) * S1_0 * norm.cdf(-d1)

    return exOP
