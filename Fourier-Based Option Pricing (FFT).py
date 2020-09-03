# Fourier-Based Option Pricing (FFT)

import math
import numpy as np
from numpy.fft import *
from scipy import stats # For Calibration
from numpy.fft import fft

# Merton (1976) Jump-Diffusion Model  
# For a European Call Option using the Fourier Based Approach

S0 = 100.0  # initial index level
K = 95.0  # strike level
T = 1.0  # call option maturity
r = 0.05  # constant short rate
sigma = 0.2  # constant volatility of diffusion
lmbda = 1.0  # jump frequency p.a.
mu = -0.1  # expected jump size
delta = 0.05  # jump size volatility

# Merton Characteristic Equation
def mertonCF(u, x0, T, r, sigma, lmbda, mu, delta):
    omega = x0 / T + r - 0.5 * sigma ** 2 \
        - lmbda * (np.exp(mu + 0.5 * delta ** 2) - 1)
    value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2 +
                    lmbda * (np.exp(1j * u * mu -
                    u ** 2 * delta ** 2 * 0.5) - 1)) * T)
    return value
# from here one could use the Lewis (2001) approch with numerical integration
# or as we will do, follow th Carr-Madan (1999) Approach and use FFT

# Carr-Madan Approach with FFT
def fft_call(S0, K, T, r, sigma, lmbda, mu, delta):
    k = math.log(K / S0)
    x0 = math.log(S0/S0)
    g = 4  # factor to increase accuracy
    N = g * 2048
    eps = (g * 150.0) ** -1
    eta = 2 * math.pi / (N * eps)
    b = 0.5 * N * eps - k
    u = np.arange(1, N + 1, 1)
    vj = eta * (u - 1)
    # Modificatons to Ensure Integrability
    if S0 >= k:  # ITM case
        alpha = 1.5
        v = vj - (alpha + 1) * 1j
        modCF = math.exp(-r * T) * mertonCF(
            v, x0, T, r, sigma, lmbda, mu, delta) / (alpha ** 2 + alpha - vj ** 2 + 
            1j * (2 * alpha + 1) * vj)
    else:  # OTM case - this has 2 modified characteristic functions since is can be dampened twice due to its symmetry, 
           # these are weighted later on
        alpha = 1.1
        v = (vj - 1j * alpha) - 1j
        modCF1 = math.exp(-r * T) * (1 / (1 + 1j * (vj - 1j * alpha))
               - math.exp(r * T) /(1j * (vj - 1j * alpha))- 
                 mertonCF(v, x0, T, r, sigma, lmbda, mu, delta) /
                 ((vj - 1j * alpha) ** 2 - 1j * (vj - 1j * alpha)))

        v = (vj + 1j * alpha) - 1j
        modCF2 = math.exp(-r * T) * (1 / (1 + 1j * (vj + 1j * alpha)) -
                 math.exp(r * T) /(1j * (vj + 1j * alpha))- 
                 mertonCF(v, x0, T, r, sigma, lmbda, mu, delta) / 
                 ((vj + 1j * alpha) ** 2 - 1j * (vj + 1j * alpha)))

    # Fast Fourier Transform
    kron_delta = np.zeros(N, dtype=np.float) # the Kronecker Delta Function is 0 for n = 1 to N
    kron_delta[0] = 1 # function takes value 1 for n = 0
    j = np.arange(1, N + 1, 1)

    # Simpsons Rule
    simpson = (3 + (-1) ** j - kron_delta) / 3

    if S0 >= 0.95 * K:
        fftfunction = np.exp(1j * b * vj) * modCF * eta * simpson
        payoff = (fft(fftfunction)).real # calls numpy's fft functions
        call_value_pre = np.exp(-alpha * k) / math.pi * payoff

    else:
        fftfunction = (np.exp(1j * b * vj) * (modCF1 - modCF2) *
                       0.5 * eta * simpson)
        payoff = (fft(fftfunction)).real
        call_value_pre = payoff / (np.sinh(alpha * k) * math.pi)

    pos = int((k + b) / eps) # position
    call_value = call_value_pre[pos]

    return call_value * S0

print(fft_call(S0, K, T, r, sigma, lmbda, mu, delta))
