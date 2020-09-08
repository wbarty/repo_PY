from pylab import mpl, plt
import scipy as sp
import scipy.stats as sm
import scipy.optimize as sco
import scipy.interpolate as sci
import scipy
import pandas as pd
import math
import numpy as np
import statsmodels.api as sm
import numpy as np


def gen_paths(S0, r, sigma, T, M, I):
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rand)
    return paths


S0 = 100.
r = 0.017
sigma = 0.2
T = 2.0
M = 1
I = 2500000
np.random.seed(1000)
paths = gen_paths(S0, r, sigma, T, M, I)
S0 * math.exp(r * T)
'''
paths[-1].mean()

plt.figure(figsize=(10, 6))

plt.plot(paths[:, :10])

plt.xlabel('time steps')

plt.ylabel('index label')

# print graph of simulation

# show normal return array
print(paths[:, 0].round(4))


log_returns = np.log(paths[1:] / paths[:-1])

# print log return array
print(log_returns[:, 0].round(4))

# print_statistics(log_returns.flatten())


# Annualised returns
log_returns.mean() * M + 0.5 * sigma**2


# Annualised volatility
log_returns.std() * math.sqrt(M)
'''


# USING RAW DATA NOT LOG RETURNS
raw = pd.read_csv(r'C:\Users\wbart\Desktop\352data\352Pricingdata.csv', index_col=0, parse_dates=True).dropna()

# print(raw)

# raw.info()  # give data type...float

(raw / raw.iloc[0] * 100).plot(figsize=(100, 60))

# plt.show()

log_returns = np.log(raw / raw.shift(1))
# print(log_returns.head())

symbols = ['Stock 1', 'Stock 2', 'Stock 3', 'Stock 4', 'Stock 5', 'Stock 6', 'Stock 7', 'Stock 8', 'Stock 9', 'Stock 10', 'Stock 11', 'Stock 12', 'Stock 13', 'Stock 14', 'Stock 15', 'Stock 16', 'Stock 17', 'Stock 18', 'Stock 19', 'Stock 20', 'Stock 21', 'Stock 22', 'Stock 23', 'Stock 24', 'Stock 25', 'Stock 26', 'Stock 27', 'Stock 28', 'Stock 29', 'Stock 30', 'Stock 31', 'Stock 32', 'Stock 33', 'Stock 34', 'Stock 35', 'Stock 36', 'Stock 37', 'Stock 38', 'Stock 39', 'Stock 40', 'Stock 41', 'Stock 42', 'Stock 43', 'Stock 44', 'Stock 45', 'Stock 46', 'Stock 47', 'Stock 48', 'Stock 49', 'Stock 50']

for sym in symbols:
    log_raw = np.array(log_returns[sym].dropna())
    print(log_raw.mean, log_raw.std, log_raw.min, log_raw.max, log_raw.size)
    # plt.show()

for sym in symbols:
    log_data = np.array(log_returns[sym].dropna())
    # normality_tests(log_data)

noa = 51
data = raw[symbols]
rets = np.log(raw / raw.shift(1))
rets.hist(bins=400, figsize=(100, 60))

# Mean and Covariance Matrix for all Shares Annualised
#print(rets.mean() * 252)
#print(rets.cov() * 252)

weights = np.random.random(noa)
weights /= np.sum(weights)

# print(weights)
# print(weights.sum())

# Annualised Portfolio return given current portfolio weights
#print(np.sum(rets.mean() * weights) * 252)

# Annualised Portfolio variance given current portfolio weights
#print(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


# Annualised Portfolio volatility / standard deviation given current portfolio weights
#print(math.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))


def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


prets = []
pvols = []
for p in range(2000):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

#print(prets, pvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols, marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')

# plt.show()

#export_csv = df.to_csv(r'C:\Users\wbart\Desktop\352data\pretspvols.csv', index=None, header=True)


def min_func_sharpe(weights):
    return -port_ret(weights) / port_vol(weights)  # add in rf rate here


cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # sum to -1

bnds = tuple((0, 1) for x in range(noa))

eweights = np.array(noa * [1. / noa, ])
# print(eweights)

# print(min_func_sharpe(eweights))

opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)

# print(opts)

# print(opts['x'].round(5))

# print(port_ret(opts['x']).round(5))
# print(port_vol(opts['x']).round(5))
#print((port_ret(opts['x']) - 0.017) / port_vol(opts['x']))

optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

# print(optv)

# print(optv['x'].round(5))

# print(port_vol(optv['x']).round(5))
# print(port_ret(optv['x']).round(5))

#print((port_ret(optv['x']) - 0.017) / port_vol(optv['x']))

cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in weights)

trets = np.linspace(0.05, 0.2, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols, marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=4.0)
plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0)
plt.plot(port_vol(opts['x']), port_ret(optv['x']), 'r*', markersize=15.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')
plt.show()

ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]

tck = sci.splrep(evols, erets)


def f(x):
    return sci.splev(x, tck, der=0)


def df(x):
    return sci.splev(x, tck, der=1)


def equations(p, rf=0.017):
    eq1 = rf - p[0]
    eq2 = rf + p[1] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


opt = sco.fsolve(equations, [0.01, 0.5, 0.15])

print(np.round(equations(opt), 6))

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=(prets - 0.017) / pvols, marker='.', cmap='coolwarm')
plt.plot(evols, erets, 'b', lw=4.0),
cx = np.linspace(0.0, 0.3),
plt.plot(cx, opt[0] + opt[1], 'r', lw=1.5),
plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0),
plt.grid(True),
plt.axhline(0, color='k', ls='--', lw=2.0),
plt.axvline(0, color='k', ls='--', lw=2.0),
plt.xlabel('expected volatility'),
plt.ylabel('expected return'),
plt.colorbar(label='Sharpe ratio'),
plt.show()
