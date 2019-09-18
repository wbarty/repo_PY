# Cox-Rubinstein Binomial Model - European Option Valuation

import math
import numpy as np

# Model Parameters
S0 =
K =
T =
r =
v =
M =

# Function


def crr(S0, K, T, r, v, option='call', M):
    dt = T / M
    df = math.exp(-r * dt)  # discounting rate per period
    # Binomial parameters
    u = math.exp(v * marth.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability
    # Index level init
    mu = np.arrange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md
    # Call/Put allocation
    if option == 'call':
        Value = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (p * V[0:M - z, t + 1] + (1 - p) * V[1:M - z + 1, t + 1]) + df
        z += 1

    return V[0, 0]
