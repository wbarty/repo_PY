# Non_dividend/Dividend-Paying Asset Greeks

import numpy as np
import scipy.stats as si
import sympy as sy
import sympy.statistics as systats

# Delta


def delta(S, K, T, r, q, v, option='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    if option == 'call':
        delta = si.norm.cdf(d1, 0.0, 1.0)

    if option == 'put':
        delta = -si.norm.cdf(-d1, 0.0, 1.0)

    return result

# Gamma


def gamma(S, K, T, q, r, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    gamma = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) / S * v * np.sqrt(T)

    return gamma

# Vega


def vega(S, S0, K, K, T, q, r, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    vega = 1 / np.sqrt(2 * np.pi) * S * np.exp(-q * T) * np.exp(-d1 ** 2 * 0.5) * np.sqrt(T)

    return vega

# Theta


def theta(S, K, T, r, q, v, option='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    if option == 'call':
        theta = -np.exp(-q * T) * (S * si.norm.cdf(d1, 0.0, 1.0) * v) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) + q * S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    if option == 'put':
        theta = -np.exp(-q * T) * (S * si.norm.cdf(d1, 0.0, 1.0) * v) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - q * S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)

    return theta

# Rho


def rho(S, K, T, r, q, v, option='call'):
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    if option == 'call':
        rho = K * T np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':
        rho = -K * T np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)

    return rho
