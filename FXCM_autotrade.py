import math
import time
import numpy as np
import pandas as pd
import datetime as dt
import cufflinks as cf
from pylab import plt
import fxcmpy
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.stats as scs
import pickle
np.random.seed(1000)

p = 0.55
f = p - (1 - p)
# print(f)
I = 50
n = 100


def run_simulation(f):
    c = np.zeros((n, I))
    c[0] = 100
    for i in range(I):
        for t in range(1, n):
            o = np.random.binomial(1, p)
            if o > 0:
                c[t, i] = (1 + f) * c[t - 1, i]
            else:
                c[t, i] = (1 - f) * c[t - 1, i]
    return c


c_1 = run_simulation(f)
# print(c_1.round(3))
'''
plt.show(
plt.figure(figsize=(10, 6))
plt.plot(c_1, 'b', lw=0.5)
plt.plot(c_1.mean(axis=1), 'r', lw=2.5))
'''

# Re-running sim with different values of f
c_2 = run_simulation(0.05)
c_3 = run_simulation(0.25)
c_4 = run_simulation(0.5)

'''
plt.figure(figsize=(10, 6))
plt.plot(c_1.mean(axis=1), 'r', label='$f^*=0.1$')
plt.plot(c_2.mean(axis=1), 'b', label='$f=0.05$')
plt.plot(c_3.mean(axis=1), 'y', label='$f=0.25$')
plt.plot(c_4.mean(axis=1), 'm', label='$f=0.5$')
plt.legend(loc=0)
plt.show()
'''

raw = pd.read_csv(r'C:\Users\wbart\Desktop\352data\352Pricingdata.csv', index_col=0, parse_dates=True)
symbol = 'Market Index'
data = pd.DataFrame(raw[symbol])
# print(data)
data['returns'] = np.log(data / data.shift(1))
data.dropna(inplace=True)
# data.tail()

mu = data.returns.mean() * 252
# print(mu)

sigma = data.returns.std() * 252**0.5
# print(sigma)

r = 0.0

# Optimal Kelly Fraction Leverage Ratio, i.e. for $1 avaliballe invest f*$1 in a long position
f = (mu - r) / sigma**2
# print(f)

equs = []


def kelly_strategy(f):
    global equs
    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)
    cap = 'capital_{:.2f}'.format(f)
    data[equ] = 1
    data[cap] = data[equ] * f
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data['returns'].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f


kelly_strategy(f * 0.5)  # half of f
kelly_strategy(f * 0.66)  # 2/3 of f
kelly_strategy(f)

'''
print(data[equs].tail())
ax = data['returns'].cumsum().apply(np.exp).plot(legend=True, figsize=(10, 6))
data[equs].plot(ax=ax, legend=True)
plt.show()
'''

TOKEN = '6aaa3ede83eb9c91f9c5e85c782b1ad0f36008ed'
# https://fxcmpy.tpq.io/
api = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')

data = api.get_candles('EUR/USD', period='m5', start='2019-03-01 00:00:00', end='2019-03-30 00:00:00')
data.iloc[-5:, 4:]

spread = (data['askclose'] - data['bidclose']).mean()
# print(spread)

data['midclose'] = (data['askclose'] + data['bidclose']) / 2

ptc = spread / data['midclose'].mean()

'''
# Plot the Proportional Transaction Costs given the Average Spread
data['midclose'].plot(figsize=(10, 6), legend=True)
plt.show()
'''

data['returns'] = np.log(data['midclose'] / data['midclose'].shift(1))
data.dropna()
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = data['returns'].shift(lag)
    cols.append(col)

data.dropna(inplace=True)
data[cols] = np.where(data[cols] > 0, 1, 0)
data['direction'] = np.where(data['returns'] > 0, 1, -1)
data[cols + ['direction']].head()


model = SVC(C=1, kernel='linear', gamma='auto')
split = int(len(data) * 0.80)
train = data.iloc[:split].copy()

model.fit(train[cols], train['direction'])
data['midclose'].plot(figsize=(10, 6), legend=True)
# plt.show()

accuracy_score(train['direction'], model.predict(train[cols]))
test = data.iloc[split:].copy()
test['position'] = model.predict(test[cols])
accuracy_score(test['direction'], test['position'])

test['strategy'] = test['position'] * test['returns']
sum(test['position'].diff() != 0)

test['strategy_tc'] = np.where(test['position'].diff() != 0, test['strategy'] - ptc, test['strategy'])
test[['returns', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize=(10, 6))
# plt.show()

mean = test[['returns', 'strategy_tc']].mean() * len(data) * 12
# print(mean)
var = test[['returns', 'strategy_tc']].var() * len(data) * 12
# print(var)
vol = var**0.5
# print(vol)

# print(mean/var)
# print(mean/var**05)

# Adjusting for different Leverage Values
to_plot = ['returns', 'strategy_tc']
for lev in [10, 20, 30, 40, 50]:
    label = 'lstrategy_tc_%d' % lev
    test[label] = test['strategy_tc'] * lev
    to_plot.append(label)

test[to_plot].cumsum().apply(np.exp).plot(figsize=(10, 6))
# plt.show()


# RISK ANALYSIS

equity = 3333
risk = pd.DataFrame(test['lstrategy_tc_30'])
risk['equity'] = risk['lstrategy_tc_30'].cumsum().apply(np.exp) * equity
risk['cummax'] = risk['equity'].cummax()
risk['drawdown'] = risk['cummax'] - risk['equity']

risk['drawdown'].max()

t_max = risk['drawdown'].idxmax()
# print(t_max)

temp = risk['drawdown'][risk['drawdown'] == 0]
periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())

periods[20:30]
# print(periods)

# t_per = periods.max()
# print(t_per)
# t_per = periods.max()

# t_per.seconds / 60 / 60

risk[['equity', 'cummax']].plot(figsize=(10, 6))
plt.axvline(t_max, c='r', alpha=0.5)
# plt.show()


percs = np.array([0.01, 0.1, 1., 2.5, 5.0, 10.0])
risk['returns'] = np.log(risk['equity'] / risk['equity'].shift(1))

VaR = scs.scoreatpercentile(equity * risk['returns'], percs)


def print_var():
    print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
    print(33 * '-')
    for pair in zip(percs, VaR):
        print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))


# print(print_var())

# Resampling the VaR from 5M to 1H basis
hourly = risk.resample('1H', label='right').last()
hourly['returns'] = np.log(hourly['equity'] / hourly['equity'].shift(1))

VaR = scs.scoreatpercentile(equity * hourly['returns'], percs)

# print(print_var())


# Persisting the Model Object

pickle.dump(model, open('algorithm.pkl', 'wb'))

# ONLINE ALGORITHM

algorithm = pickle.load(open('algorithm.pkl', 'rb'))
sel = ['tradeId', 'amountL', 'currency', 'grossPL', 'isBuy']


def print_positions(pos):
    print('\n\n' + 50 * '=')
    print('Going {}.\n'.format(pos))
    time.sleep(1.5)
    print(api.get_open_positions()[sell])
    print(50 * '=' + '\n\n')


symbol = 'EUR/USD'
bar = '15s'
amount = 100
postion = 0
min_bars = lags + 1
df = pd.DataFrame()


def automated_strategy(data, dataframe):
    global min_bars, position, df
    ldf = len(dataframe)
    df = dataframe.resample(bar, label='right').last().ffill()
    if ldf % 20 == 0:
        print('%3d' % len(dataframe), end=',')

    if len(df) > min_bars:
        min_bars = len(df)
        df['Mid'] = df = df[['Bid', 'Ask']].mean(axis=1)
        df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
        df['Direction'] = np.where(df['Returns'] > 0, 1, -1)
        features = df['Direction'].iloc[-(lags + 1):-1]
        features = features.values.reshape(1, -1)
        signal = algorithm.predict(features)[0]

        if position in [0, -1] and signal == 1:
            api.create_market_buy_order(symbol, amount - position * amount)
            position = 1
            print_positions('LONG')
        elif position in [0, 1] and signal == -1:
            api.create_market_sell_order(symbol, amount + position * amount)
            position = -1
            print_positions('SHORT')
        if len(dataframe) > 350:
            api.unsubscribe_market_data('EUR/USD')
            api.close_all()
