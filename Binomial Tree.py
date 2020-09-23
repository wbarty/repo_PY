import math
import numpy as np

S0 = 100
T = 1
r = 0.05
sigma= 0.2
N = 5

def simulate_tree(M):
     dt = T/M
     u = math.exp(sigma*math.sqrt(dt))
     d=1/u
     S = np.zeros((M+1, M+1))
     S[0,0] = S0
     z=1
     for t in range(1,M+1):
             for i in range(z):
                     S[i,t] = S[i,t-1]*u
                     S[i+1,t] = S[i,t-1]*d
             z += 1
     return S

simulate_tree(N)
