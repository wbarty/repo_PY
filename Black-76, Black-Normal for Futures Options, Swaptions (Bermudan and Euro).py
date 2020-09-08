# Black-76 model for valuing Futures Options, Swaptions with the Black-Normal Model

import numpy as np
import pandas as pd
import scipy.stats

# Black-76, is similar to BSM however the Spot price S(t) is replaced by a discounted factor futures price F
# The B76 assumes log-normality with futures price F(t) and constant risk free rate r

F =  # futures price
K =
r =
T =
v =  # volatility


def b76(F, K, r, T, v, option='call'):
    d1 = d1 = (np.log(F / K)) + ((v**2 / 2) * T) / np.sqrt(v) * np.sqrt(T)
    d2 = d1 - np.sqrt(v) * np.sqrt(T)

    if option == 'call':
        b76 = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if option == 'put':
        b76 = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return b76


# European Swaption using B76
C =  # forward swap rate between n and N
m =  # the number of reset days per year, when floating rate resets at its fixed intervals
tau =  # tenor - time between swaption and swap maturities
annuity_factor = (1 - 1 / (1 + C / m)**(tau * m)) / C


def b76_swaption(C, K, r, m, tau, r, T, v, annuity_factor, swap='PS'):  # pays the fixed rate PS, receiver RS
    d1 = (np.log(C / K) + 0.5 * v**2 * T) / v * np.sqrt(T)
    d2 = d1 - v * np.sqrt(T)

    if swap == 'PS':
        b76_swaption = annuity_factor * np.exp(-r * T) * (C * norm.cdf(d1) - K * norm.cdf(d2))
    if swap == 'RS':
        b76_swaption = annuity_factor * np.exp(-r * T) * (K * norm.cdf(-d2) - C * norm.cdf(-d1))

    return b76_swaption


# Black-Normal Model - allows for negative rates and thus normality in the volatility data
# BN Swaption value


# need vN which is normal volatility for this calculation, to use this we also need vB the implied volatility from BSM
n = norm.pdf
N = norm.cdf

# BS price


def bs_price(S, K, T, r, v, q, option='call'):
    d1 = (log(S / K) + (r + v * v / 2.) * T) / (v * sqrt(T))
    d2 = d1 - v * sqrt(T)
    if option == 'call':
        price = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
    else:
        price = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)
    return price

# Vega


def bs_vega(S, K, T, r, v, q, option='call'):
    d1 = (log(S / K) + (r + v * v / 2) * T) / (v * sqrt(T))
    return S * sqrt(T) * n(d1)


# Newtons method for finding implied volatility
call_put = call

def impvol(target_value, call_put, S, K, T, r):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5
    for i in xrange(0, MAX_ITERATIONS):
        price = bs_price(call_put, S, K, T, r, sigma)
        vega = bs_vega(call_put, S, K, T, r, sigma)

        price = price
        diff = target_value - price  # our root

        print i, sigma, diff

        if (abs(diff) < PRECISION):
            return sigma
        vB = sigma + diff / vega  # f(x) / f'(x)

    return vB


# Normal Volatility Transformation
vN = C * np.sqrt(2 * np.pi / T) * (2 * norm.cdf((vB * np.sqrt(T) / 2) - 1))

# Black-Normal Swaption Calculation


def bn_swaption(C, K, r, m, tau, r, T, vN, annuity_factor, swap='PS'):
    d = (C - K) / (vN * np.sqrt(T))

    if swap == 'PS':
        bn_swaption = annuity_factor * np.exp(-r * T) * ((C - K) * norm.cdf(d) + (vN * np.sqrt(T)) / np.sqrt(2 * np.pi) * np.exp(-d**2 / 2))
    if swap == 'RS':
        bn_swaption = annuity_factor * np.exp(-r * T) * ((K - C) * norm.cdf(d) + (vN * np.sqrt(T)) / np.sqrt(2 * np.pi) * np.exp(-d**2 / 2))

    return bn_swaption
