# Non_dividend/Dividend-Paying Asset Greeks

import numpy as np
import scipy.stats as si
import sympy as sy
import sympy.statistics as systats

data = pd.read_csv('')

######################################
# First Order Greeks (incl. Gamma)
######################################

# Delta


def delta(S, K, T, r, q, v, option='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    if option == 'call':
        delta = si.norm.cdf(d1, 0.0, 1.0)

    if option == 'put':
        delta = -si.norm.cdf(-d1, 0.0, 1.0)

    return result

# print(data.apply(delta))

# Gamma


def gamma(S, K, T, q, r, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    gamma = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) / S * v * np.sqrt(T)

    return gamma

# print(data.apply(gamma))

# Vega


def vega(S, S0, K, K, T, q, r, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    vega = 1 / np.sqrt(2 * np.pi) * S * np.exp(-q * T) * np.exp(-d1 ** 2 * 0.5) * np.sqrt(T)

    return vega

# print(data.apply(vega))

# Theta


def theta(S, K, T, r, q, v, option='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    if option == 'call':
        theta = -np.exp(-q * T) * (S * si.norm.cdf(d1, 0.0, 1.0) * v) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) + q * S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    if option == 'put':
        theta = -np.exp(-q * T) * (S * si.norm.cdf(d1, 0.0, 1.0) * v) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - q * S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)

    return theta

# print(data.apply(theta))


# Rho


def rho(S, K, T, r, q, v, option='call'):
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    if option == 'call':
        rho = K * T np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':
        rho = -K * T np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)

    return rho

# print(data.apply(rho))

# Lambda - percentage change in option prices per percentage change in underlying price, a measure of leverage/gearing - omega/elasticity


def lamb_da(S, K, T, r, q, v, option='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

        def delta(S, K, T, r, q, v, option='call'):
        d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
            if option == 'call':
                delta = si.norm.cdf(d1, 0.0, 1.0)
            if option == 'put':
                delta = -si.norm.cdf(-d1, 0.0, 1.0)
            return delta

            def BSMvalue(S, K, T, r, q, v, option == 'call'):
                if option == 'call':
                    BSMvalue = S * delta - e ** (-r * T) * K * norm.cdf(d2)
                if option == 'put':
                    BSMvalue = e ** (-r * T) * K * norm.cdf(-d2) - S * -delta
                return BSMvalue

        lamb_da = delta * (S / BSMvalue)

######################################
# Second Order Greeks
######################################

# Vanna - measures assets Delta sensitivity to changes in the underlying volatility or vega's sensitivity to a change in S
    # derived from vega


def vanna(S, K, r, q, T, v, cvol):  # cvol = change in volatility
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    vanna = [cvol * -e ** (-q * T) * d2 / v * norm.pdf(d1)]

    return vanna

# Charm - change in delta for a change in time or theta's sensitivity for a change in S
    # derived from Theta - DdeltaDtime


def charm(S, K, r, q, T, v, timechange, option == 'call'):  # timechange should be in days/365 e.g. 1/365
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    dfq = e ** (-q * T)
    if option = 'call':
        charm = timechange * -dfq * (norm.pdf(d1) * ((r - q) / (v * T) - d2 / (2 * T)) + (-q) * norm.cdf(d1))
    if option == 'put':
        charm = timechange * -dfq * (norm.pdf(d1) * ((r - q) / (v * T) - d2 / (2 * T)) + q * norm.cdf(-d1))

    return charm

# Vomma - sensitivity of vega to changes in volaility - DvegaDvol


def vomma(S, K, r, q, T, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    def vega(S, S0, K, K, T, q, r, v):
        d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
        vega = 1 / np.sqrt(2 * np.pi) * S * np.exp(-q * T) * np.exp(-d1 ** 2 * 0.5) * np.sqrt(T)

    vomma = vega * ((d1 * d2) / v)

    return vomma

# Veta - measure the rate of change in the vega wrt the passage of time - DvegaDtime


def vomma(S, S0, K, r, q, T, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    def vega(S, S0, K, K, T, q, r, v):
        d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
        vega = 1 / np.sqrt(2 * np.pi) * S * np.exp(-q * T) * np.exp(-d1 ** 2 * 0.5) * np.sqrt(T)

    veta = -vega * (q + ((r - q) * d1) / v * sqrt(T) - ((1 + d1 * d2) / 2 * T))

    return veta

# Vera - measures the rate of change in rho wrt volatility - rhova
##################################################################

######################################
# Third Order Greeks
######################################

# Speed - measures the rate of change in Gamma wrt changes in the underlying prices - DgammaDspot (gamma of gamma)


def speed(S, K, r, q, T, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    def gamma2(S, K, r, q, T, v):
        gamma = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) / (S ** 2 * v * np.sqrt(T))  # changes S to S^2

    speed = -gamma2 * (d1 / v * sqrt(T) + 1)

    return speed

# Zomma - measures the rate of change of gamma wrt changes in volatility - DgammaDvol


def zomma(S, K, r, q, T, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    def gamma2(S, K, r, q, T, v):
        gamma = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) / (S ** 2 * v * np.sqrt(T))  # changes S to S^2

    zomma = gamma2 * (d1 * d2 - 1) / S * v ** 2 * sqrt(T)

    return zomma

# Color - measures the rate of change of gamma over the passage of time - DgammaDtime (gamma decay)


def color(S, K, r, q, T, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    color = -e ** (-q * T) * ((norm.cdf(d1)) / 2 * S * T * v * sqrt(T)) * (2 * q * T + 1 + ((2 * (r - q) * T - d2 * v * sqrt(T)) / v * sqrt(T)) * d1)

    return color

# Ultima - measures the sensitivity of the option to vomma wrt change in volatility - DvommaDvol

 def ultima(S, K, r, q, T, v):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    def vega(S, S0, K, K, T, q, r, v):
        d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
        vega = 1 / np.sqrt(2 * np.pi) * S * np.exp(-q * T) * np.exp(-d1 ** 2 * 0.5) * np.sqrt(T)

    ultima = (-vega / v ** 2) * (d1 * d1 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)

    return ultima


