# BCC97 Market Model Code
# web post: https://qfaus.com/2020/09/22/bcc97/
import math
import numpy as np
np.seterr(all='print')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize

#######################
##### CALIBRATION #####
#######################

# CIR85 Calibration
## First we set up the framework for valuing a ZCB

### Dummy values
r0, kappa_r, theta_r, sigma_r, t, T = 0.1, 0.1, 0.1, 0.1, 0.1, 5
alpha = r0, kappa_r, theta_r, sigma_r, t, T # setting parameters to alpha variable for simplicity

def gamma(kappa_r, sigma_r):
    return np.sqrt(kappa_r**2 + 2*sigma_r**2)

def a1(alpha): # this is a(t,T)
    g = gamma(kappa_r, sigma_r)
    return ((2*g*np.exp((kappa_r+g)*(T-t)/2))/2*g+(kappa_r+g)*(np.exp(g*(T-t))-1))*np.exp(2*kappa_r*theta_r/sigma_r**2)

def b1(alpha): # b(t,T)
    g = gamma(kappa_r, sigma_r)
    return ((2*g*np.exp(g**(T-t)-1)))/(2*g+(kappa_r+g)*(np.exp(g*(T-t))-1))

def B(alpha): # B(T), the formula for ZCB under CIR85
    E_rt = theta_r+np.exp(kappa_r*t)*(r0 - theta_r) # E(rt) expectation of rt
    zcb = a1(alpha)*np.exp(-b1(alpha)*E_rt)
    return zcb

## Setting up the data for Calibration
### Rates are the BBSW from 30-day to 1-year
r_rates = np.array([0.09, 0.09, 0.14, 0.11])/100 # in decimal terms not %
### https://www.yieldreport.com.au/category/bank-billswaps/weekly-bank-billswaps/
### Term to Maturity for each rate
t_rates = np.array([30, 90, 180, 360])/360 # assuming 360-day conventions
### Quantlib has several indexes that a user could implment however none for Australia

zero_rates = t_rates*np.log(1+(r_rates/t_rates)) # conversion to continuous rates

### set r0 to new value
r0 = r_rates[0]

## Interpolate Data using Cubic Splines (an interpolation technique)
tck = interpolate.splrep(t_rates, zero_rates)
### scipy requires variables T, C, and K, here T (knot-points) is maturities, C (coefficients)
### is the zero rates and we leave K blank which gives the spline order, without an input the default
### is cubic, we can also speicy s = - for a given level of smoothness, default is 3
tn_list = np.linspace(0.0, 1.0, 24)
ts_list = interpolate.splev(tn_list, tck, der=0) # splev is the spline,
de_list = interpolate.splev(tn_list, tck ,der=1) # der specifies the derivative

f = ts_list + de_list + tn_list # this complete the forward rate interpolation and transformation

def ts_plot(): # plotting the original term structure vs the interpolated vs the derivative
               # derivative shows when the original rates are below the interpolation
    plt.figure(figsize=(8,6))
    plt.plot(t_rates, r_rates, label = 'rates')
    plt.plot(tn_list, ts_list, label = 'interpolation', lw = 0.5)
    plt.plot(tn_list, de_list, label = '1st derivative', lw = 1.5)
    plt.legend(loc=0)
    plt.xlabel('Time (Years)')
    plt.ylabel('Rate')
    plt.grid()
    #plt.show()
# ts_plot()

## Model the Forward Rates for CIR85 under HJM (summed over t;T)
opt = kappa_r, theta_r, sigma_r # these are used from here and into the optimistion
def CIRfr(opt):
    t = tn_list
    g = gamma(kappa_r, sigma_r)
    sum1 = ((kappa_r*theta_r*(np.exp(g*t)-1))/(2*g+(kappa_r+g)*(np.exp(g*t)-1)))
    sum2 = r0*((4*g**2*np.exp(g*t))/(2*g+(kappa_r+g)*(np.exp(g*t)-1))**2)
    fr = sum1+sum2
    return fr

## Define an error function for optimisation to minimise
def CIRerr(opt):
    #if 2*kappa_r*theta_r < sigma_r**2:
        #return 100
   # elif kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
       # return 100
    forward_rates = CIRfr(opt)
    MSE = np.sum((f - forward_rates)**2/len(f)) # remember f = ts+tn+de lists
    return MSE
# print(CIRerr(opt))
### returned 0.3372863680606061

## Calibration
def CIRcal():
    opt = optimize.minimize(CIRerr, method='Powell', x0 = [1.00, 0.02, 0.1]) # minimising CIRerr, array of initial guesses
    return opt # uses some minimisation algorithm in scipy, gives an array of optimised parameter values for kappa, theta and sigma
# print(CIRcal())


# Heston 1993 Model Calibration
import pandas as pd
import datetime as dt
# reuses the 'B' function from CIR85 calibration

## Requires calibrated CIR85 values for the parameters
kappa_r = 3.58792896
theta_r = 2.60792896
sigma_r = 2.68792896

# Initial values for H93 parameters
kappa_v = 0.01
theta_v = 0.05
sigma_v = 0.02
rho = 0.5
v0 = 0.01

data = pd.read_csv(r'C:\Users\wbart\OneDrive\Documents\Programming\Python\BCC97\optionsdata.csv', index_col=2, header=0)
# print(data)
#             Call   Put   Strike    Ticker  TTM
# Maturity
# 15/10/2020  1.55  1.18    5900     XJOL37   28
# 19/11/2020  1.99  2.27    5950     XJOLJ7   63
# 17/12/2020  2.28  2.81    5975     XJO0Q8   91

S0 = 5946.200 # S&P/ASX200 as of 2020-09-17
r0 = r_rates[0] # starting interest rate
t0 = dt.date(2020,9,17)

o = 0.05 # a set % of Out of moneyness/in the moneyness for option selection
options = data[np.abs(data['Strike']-S0)/S0 < o] # selects the appropriate options
options = data[data['Strike'].isin([5850, 5900, 5925, 5950, 5975, 6000])]

## Loops over data to add short rates
# for row, option in options.iterrows():
#     T = options['TTM']
#     B0T = B([kappa_r, theta_r, sigma_r, r0, T])
#     options[row, 'r'] = -math.log(B0T)/T

## H93 via Fourier Transform Lewis (2001)
def H93charfunc(u, T, r, kappa_v, theta_v, theta_, sigma_v, rho, v0):
    c1 = kappa_v*theta_v
    c2 = -np.sqrt((rho*sigma_v*u*1j - kappa_v**2-sigma_v**2*(-u*1j - u**2)))
    c3 = (kappa_v-rho*sigma_v*u*1j+c2)
    H1 = (r*u*1j*T+(c1/sigma_v**2)*((kappa_v-rho*sigma_r*u*1j+c2)*T - 2*np.log((1-c3*np.exp(c2*T))/(1-c3))))
    H2 = ((kappa_v - rho*sigma_v*u*1j+c2)/sigma_v**2*((1-np.exp(c2*T))/(1-c3*np.exp(c2*T))))
    charfuncvalue = np.exp(H1+H2*v0)
    return charfuncvalue

def H93intfunc(u, S0, K, T,r, kappa_v, theta_v, sigma_v, rho, v0):
    charfuncvalue = H93charfunc(u-1j*0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    int_func_value = 1/(u**2+0.25)*np.exp(1j*np.log(S0/K))*charfuncvalue.real
    return int_func_value

## H93 Call Value
def H93call(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    int_value = quad(lambda u: H93intfunc(u,S0,K,T,r,kappa_v,theta_v, sigma_v, rho,v0), 0, np.inf, limit=250)[0]
    callvalue = max(0, S0 - np.exp(-r*T)*np.sqrt(S0*K)/np.pi*int_value)
    return callvalue

## Calibration of H93
i=0
min_mse = 500
def H93err(kappa_v, theta_v, sigma_v, rho, v0):
    global i, min_mse
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or rho < -1.0 or rho > 1.0:
        return 500.00
    if 2* kappa_v * theta_v < sigma_v **2:
        return 500.00
    se = []
    for row, option in options.iterrows():
        model_value = H93call(S0, option['Strike'], option['T'], option['r'], kappa_v, theta_v, sigma_v, rho, v0)
        se.append((model_value - option['Call'])**2)
    mse = sum(se)/len(se)
    min_mse = min(min_mse, mse)
    if i % 25 ==0:
        print(np.array(kappa_v, theta_v, sigma_v, rho, v0), (mse, min_mse))
    i+=1
    return mse

def H93cal():
    p0 = optimize.brute(H93err, ((2.0, 5.0, 10.0),
                        (0.01, 0.05, 0.01),
                        (0.05, 0.25, 0.1),
                        (-0.75, 0.25, 0.1),
                        (0.01, 0.03, 0.01)), finish='none')
    opt = optimize.fmin(H93err, p0, xtol=0.000001, ftol=0.000001, maxiter=750, maxfun=900)
    return opt
print(H93cal())
def H93modelval(kappa_v, theta_v, sigma_v, rho, v0):
    values=[]
    for row, option in options.iterrows():
        model_value = H93call(S0, option['Strike'], option['T'], option['r'], kappa_v, theta_v, sigma_v, rho, v0)
        values.append(model_value)
    return np.array(values)
print(H93modelval(kappa_v, theta_v, sigma_v, rho, v0))
# M76 Jump-Diffusion Component
## Requires calibrated CIR85 values for the parameters
S0 = 5946.200 # S&P/ASX200 as of 2020-09-17
r0 = r_rates[0] # starting interest rate

## M76 via Fourier Transform
def M76charfunc(u,T,r,sigma,lamb,mu,delta):
    omega = r-0.5*sigma**2-lamb*(np.exp(mu+0.5*delta**2)-1)
    charfuncvalue = np.exp((1j*u*omega-0.5*u**2*sigma**2 + lamb*(np.exp(1j*u*mu-u**2*delta**2*0.5)-1))*T)
    return charfuncvalue

def M76intfunc(u, S0, K, T, r, sigma, lamb,mu,delta):
    charfuncvalue = M76charfunc(u-0.5*1j,T,r,sigma,lamb,mu,delta)
    intfuncvalue = 1/(u**2+0.25)*(np.exp(1j*u*np.log(S0/K))*charfuncvalue).real
    return intfuncvalue

def M76call(S0, K, T, r, v0, lamb, mu, delta):
    sigma = np.sqrt(v0)
    intvalue = quad(lambda u: M76intfunc(u, S0, Kt, r, sigma, lamb,mu,delta), 0, np.inf, limit=250)[0]
    callvalue = max(0, S0-np.exp(-r*T)*np.sqrt(S0*K)/np.pi*intvalue)
    return callvalue

## Calibration Functions
p0 = lamb, mu, delta
i=0
min_mse=5000.0
local_opt = False
def M76err(p0):
    global i, min_mse, local_opt, opt1
    if lamb < 0.0 or mu < -0.6 or mu >0.0 or delta < 0.0:
        return 5000.0
    se=[]
    for row, option in options.iterrows():
        model_value = M76call(S0, option['Strike'], option['T'], option['r'], kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
        mse = sum(se)/len(se)
        if i % 25==0:
            print(np.array(p0), mse, min_mse)
        i+=1
        if local_opt:
            penalty = np.sqrt(np.sum((p0 - opt1)**2))*1
            return mse+penalty
        return mse

def M76cal():
    opt1 = 0.0
    opt1 = optimize.brute(M76err,
                 ((0.0, 0.5, 0.1),
                 (-0.5, -0.11, 0.1),
                 (0.0, 0.5, 0.25)),
                 finish = none)
    local_opt = True
    opt2 = optimize.fmin(H93err, p0, xtol=0.0000001, ftol=0.0000001, maxiter=750, maxfun=900)
    return np.array(opt2)

def M76modelval(p0):
    values=[]
    for row, option in options.iterrows():
        T = (option['Maturity']-option['TTM'])
        B0T = B([kappa_r, theta_r, sigma_r, r0, T])
        r = -math.log(B0T/T)
        modelvalue = M76call(S0, option['Strike'], T, r, v0, lamb, mu, delta)
        values.append(modelvalue)
    return np.array(values)

# BCC97 Complete Calibration

## BCC97 Characteristic Function
def BCC79charfunc(u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    BCC1 = H93charfunc(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    BCC2 = M76charfunc(u, T, lamb, mu, delta)
    return BCC1*BCC2

## BCC97 Integration Function
def BCC97intfunc(S0, K, u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    charfuncvalue = BCC79charfunc(u-1j*0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
    intfuncvalue = 1/(u**2+0.25)*(np.exp(1j*u*np.log(S0/K))*charfuncvalue).real
    return intfuncvalue

## BCC97 Call Value
def BCC97call(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    intvalue = quad(lambda u: BCC97intfunc(S0, K, u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta), 0, np.inf, limit=250)[0]
    callvalue = max(0, S0 - np.exp(-r*T)*np.sqrt(S0*K)/np.pi*intvalue)
    return callvalue

# This model again requires the calibrated CIR85 short rates and the paramters from the H93 and M76 calibations also
kappa_r, theta_r, sigma_r = 3.58792896, 2.60792896, 2.68792896
kappa_v, theta_v, sigma_v, rho, v0 = H93cal()
lamb, mu, delta = M76cal()
p0 = [kappa_v, theta_v, sigma_v, rho, v0, lamb , mu, delta]

S0 = 5946.200
r0 = r_rates[0]

## Calibration Setup
i=0
min_mse = 5000.00

## Error Function
def BCC97err(p0):
    global i, min_mse
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or rho < -1.0 or rho > 1.0 or v0 < 0.0 or lamb < 0.0 or mu < -0.6 or mu > 0.0 or delta < 0.0:
        return 5000.00
    if 2 * kappa_v * theta_v < sigma_v**2:
        return 5000.00
    se = []
    for row, option in options.iterrows():
        model_value = BCC97call(S0, option['Strike'], option['T'], option['r'], kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
        se.append((model_value - option['Call'])**2)
    mse = sum(sex)/len(se)
    min_mse = min(min_mse, mse)
    if i % 25 == 0:
        print(np.array(p0), mse, min_mse)
    i += 1
    return mse

# Minimising the error function to calibrate
def BCC97cal():
    opt = optimize.fmin(BCC97err, p0, xtol=0.000001, ftol=0.000001, maxiter=750, maxfun=900)
    np.save('BCC97 Calibrated', np.array(opt))
    return opt

# Calculating Model Values for each parameter
def BCC97modelval(p0):
    values = []
    for row, option in options.iterrows():
        model_value = BCC97call(S0, option['Strike'], option['T'], option['r'], kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
        values.append(model_value)
    return np.array(values)

######################
##### SIMULATION #####
######################

# Simulating BCC97
T = 1.0 # time horizon
M = 20 # time steps
I = 1000 # number of replications per valuation path
anti_paths = True # variance reduction tecniques
np.random.seed(1234) # seed for RN generation

params = kappa_v, theta_v, sigma_v, rho, v0, lamb , mu, delta, r0

## Random Number Generator
def cholesky(rho): # rho is the correlation between the asset level and variance
    rho_rs = 0 # correlation between asset level and short rate
    covar = np.zeros((4,4), dtype=np.float)
    covar[0] = [1.0, rho_rs, 0.0, 0.0]
    covar[1] = [rho_rs, 1.0, rho, 0.0]
    covar[2] = [0.0, rho, 1.0, 0.0]
    covar[3] = [0.0, 0.0, 0.0, 1.0]
    cholesky_matrix = np.linalg.cholesky(covar)
    return cholesky_matrix
# Cholesky decomposition is used to decompose a positive definite matrix (such as a covariance matrix) into
#                        the product of a lower triangular matrix and its conjugate transpose (A = LL*) which is useful
#                        for numerical simulations which is basically all you need to know.

### RNG with variance reduction techniques
def rng(M, I, anti_paths): # will generate normally distributed random values for MCS
    if anti_paths:
        rand = np.random.standard_normal((4,M+1,I/2))
        rand = np.concatenate((rand, -rand),2) # joins rand and -rand
    else:
        rand = np.random.standard_normal((4, M+1,I))
    return rand

# Short rate and volatlity processes (both CIR and Heston are square root diffusion processes therefore we can use the same process with different parameters)
def squarerootdiff(x0, kappa, theta, sigma, T, M, I, rand, row, cholesky_matrix):
    dt = T/M
    x = np.zeros((M+1,I), dtype=np.float)
    x[0] = x0
    xh=np.zeroes_like(x)
    xh[0]=x0sdt=math.sqrt(dt)
    for t in range(1, M+1):
        ran = np.dot(cholesky_matrix, rand[:,t])
        xh[t] = (xh[t-1]+kappa*(theta-np.maximum(0, xh[t-1]))*dt+np.sqrt(np.maximum(0,xh[t-1]))*simga*ran[row]*sdt)
    x[t] = np.maximum(0, xh[t])
    return x

# Jump Diffusion B96 Model Index Process
def b96(S0, r, v, lamb, mu, delta, rand, row1, row2, cholesky_matrix, T, M, I):
    S = np.zeros((M+1<I), dtype=np.float)
    S[0]=S0
    dt = T/M
    sdt = math.sqrt(dt)
    ranp = np.random.poisson(lamb*dt, (M+1,I))
    rj = lamb*(math.exp(mu+0.5*delta**2)-1)
    bias=0.00
    for t in xrange(1,M+1,1):
        ran = np.dot(cholesky_matrix, rand[:,t,:])
        S[t] = S[t-1]*(np.exp(((r[t]+r[t-1])/2-0.5*v[t])*dt+np.sqrt(v[t])*ran[row1]*sdt-bias)+(np.exp(mu+delta*ran[row2])-1)*ranp[t])
    return S

# Full Model
if __name__ == '__main__':
    cholesky_matrix = cholesky(rho)
    rand = rng(M, I, anti_paths)
    sr = squarerootdiff(r0, kappa_r, theta_r, sigma_r, T, M, I, rand, 0, cholesky_matrix)
    sv = squarerootdiff(v0, kappa_v, theta_v, sigma_v, T, M, I, rand, 2, cholesky_matrix)
    S = b96(S0, sr, sv, lamb, mu, delta, rand, rand, 1, 3, cholesky_matrix, T, M, I)


# European Call Option
t_list = data['Maturity']
k_list = data['Strike']

def eurovalue(M0=50, I=50000):
    results=[]
    for T in t_list:
        M=int(M0*T)
        cholesky_matrix = cholesky(rho)
        rand = rng(M,I, anti_paths)
        sr = squarerootdiff(r0, kappa_r, theta_r, sigma_r, T, M, I, rand, 0, cholesky_matrix)
        sv = squarerootdiff(r0, kappa_v, theta_v, sigma_v, T, M, I, rand, 2, cholesky_matrix)
        S = b96(S0, sr, sv, lamb, mu, delta, rand, 1, 3, cholesky_matrix, T, M, I)
        for K in k_list:
            h = np.maximum(S[-1] - K,0)
            B0T = B([r0, kappa_r, theta_r, sigma_r, 0.00, T])
            v0_mcs = B0T*np.sum(h)/I

            ra = -math.log(B0T)/T # mean short rate
            c0 = BCC97call(S0, K, T, ra, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta) # call value
            results.append((T, K, c0, v0_mcs, v0_mcs-c0)) # comparison of each strike and maturity call values
    return results


