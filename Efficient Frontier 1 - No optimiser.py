from pylab import mpl, plt
import scipy as sp
import scipy.stats as sm
import scipy
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


def print_statistics(array):
    sta = scs.describe(array)
    print('%14s %15s' % ('statistic', 'value'))
    print(30 * '-')
    print('%14s %15.5f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))


# print_statistics(log_returns.flatten())


# Annualised returns
log_returns.mean() * M + 0.5 * sigma**2


# Annualised volatility
log_returns.std() * math.sqrt(M)
'''


# USING RAW DATA NOT LOG RETURNS
raw = pd.read_csv(r'C:\Users\wbart\Desktop\352data\352Pricingdata.csv', index_col=0, parse_dates=True).dropna()

print(raw)

raw.info()  # give data type...float

(raw / raw.iloc[0] * 100).plot(figsize=(100, 60))

# plt.show()


log_returns = np.log(raw / raw.shift(1))
print(log_returns.head())

symbols = ['Stock 1', 'Stock 2', 'Stock 3', 'Stock 4', 'Stock 5', 'Stock 6', 'Stock 7', 'Stock 8', 'Stock 9', 'Stock 10', 'Stock 11', 'Stock 12', 'Stock 13', 'Stock 14', 'Stock 15', 'Stock 16', 'Stock 17', 'Stock 18', 'Stock 19', 'Stock 20', 'Stock 21', 'Stock 22', 'Stock 23', 'Stock 24', 'Stock 25', 'Stock 26', 'Stock 27', 'Stock 28', 'Stock 29', 'Stock 30', 'Stock 31', 'Stock 32', 'Stock 33', 'Stock 34', 'Stock 35', 'Stock 36', 'Stock 37', 'Stock 38', 'Stock 39', 'Stock 40', 'Stock 41', 'Stock 42', 'Stock 43', 'Stock 44', 'Stock 45', 'Stock 46', 'Stock 47', 'Stock 48', 'Stock 49', 'Stock 50']

for(i=1;i < 51;i + +)
    a = 'Stock '

    res = a + i
    symbols.append(res)

for sym in symbols:
    print('\nresults for symbol {}'.format(sym))
    print(50 * '-')
    log_raw = np.array(log_returns[sym].dropna())
    print(log_raw.mean, log_raw.std, log_raw.min, log_raw.max, log_raw.size)
    # plt.show()

for sym in symbols:
    print('\nResults for symbol{}'.format(sym))
    print(32 * '-')
    log_data = np.array(log_returns[sym].dropna())
    # normality_tests(log_data)

noa = 51
data = raw[symbols]
rets = np.log(raw / raw.shift(1))
rets.hist(bins=400, figsize=(100, 60))

# Mean and Covariance Matrix for all Shares Annualised
print(rets.mean() * 252)
print(rets.cov() * 252)

weights = np.random.random(noa)
weights /= np.sum(weights)

print(weights)
print(weights.sum())

# Annualised Portfolio return given current portfolio weights
print(np.sum(rets.mean() * weights) * 252)

# Annualised Portfolio variance given current portfolio weights
print(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


# Annualised Portfolio volatility / standard deviation given current portfolio weights
print(math.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))


def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


prets = []
pvols = []
for p in range(20000):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

print(prets, pvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols, marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')

plt.show()

#export_csv = df.to_csv(r'C:\Users\wbart\Desktop\352data\pretspvols.csv', index=None, header=True)
