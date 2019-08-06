import scipy.stats as scs
import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl

S0 = 100.
r = 0.05
v0 = 0.1
kappa = 3.0
theta = 0.25
sigma = 0.1
rho = 0.6
T = 1.0

corr_mat = np.zeros((2, 2))
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat)

cho_mat
array([[1., 0.],
       [0.6, 0.8]])
