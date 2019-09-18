# BSM Implied Volatilities Calculation

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# use these columns from data source
cols = ['Date', '...', '...']
# pricing data
data = pd.read_csv('...',
                   header=none,
                   index_col=0,  # index at date column
                   parse_dates=T,
                   dayfirst=T / F,  # set based on date style
                   skiprows=...,  # skip irrelevant rows
                   sep=',',  # comma separated for re-csv exporting
                   names=cols)

# remove unnecessary rows
del data['...']
# set BSM variables
S0 = data['...']['sim starting date']
r = 0.014

# option data
opdata = pd.HDFStore('...')['...']

# Implied Volatility - for euro-call given the moneyness tolerance level


def impvol(opdata):
    opdata['Imp_vol'] = 0.00
    tolerance = 0.3
    for row in opdata.index:
        t = opdata['Date'][row]
        T = opdata['Maturity'][row]
        TTM = (T - t).days / 365.
        forward = np.exp(r * TTM) * S0
        if (abs(opdata['Strike'][row] - forward) / forwards) < tolerance:
            call = call_option(S0, opdata['Strike'][row], t, T, r, 0.2)
            opdata['Imp_Vol'][row] = call.imp_vol(data['Call'][row])

    return opdata
