import numpy
import algopy
import os
from algopy import UTPM, exp, Function, CGraph, zeros

# The purpose of AlgoPy is the evaluation of higher-order derivatives in the forward and reverse mode of Algorithmic Differentiation (AD) of functions that are implemented as Python programs.Particular focus are functions that contain numerical linear algebra functions as they often appear in statistically motivated functions. The intended use of AlgoPy is for easy prototyping at reasonable execution speeds. More precisely, for a typical program a directional derivative takes order 10 times as much time as time as the function evaluation.This is approximately also true for the gradient.

# utp = univariate taylor polynomial

# What can AlgoPy do -
# evaluation of derivatives useful for nonlinear continuous optimization:
#     gradient
#     Jacobian
#     Hessian
#     Jacobian vector product
#     vector Jacobian product
#     Hessian vector product
#     vector Hessian vector product
#     higher-order tensors

# Taylor series evaluation:
#     for modeling higher-order processes
#     could be used to compute Taylor series expansions for ODE/DAE integration.

# Forward Mode AD: the independant variable is fixed wrt which differentiation is performed and compute the derivative of each sub-expression recirsively - think of it as the normal order of using the chain rule.
# Reverse Mode AD: the dependant varible to be differentiated is fixed and the derivative is computed wrt each sub-expression recursively. There is only one value required to sweep for the derivative computation.

#
# Basic Derivative Calculations
#


def d_f(x):
    """function"""
    return x[0] * x[1] * x[2] + exp(x[0]) * x[1]  # x[differnce]

    # forward AD without building a computational graph
    x = UTPM.init_jacobian([3, 5, 7])
    y = d_f(x)
    algopy_jacobian = UTPM.extract_jacobian(y)
    print('jacobian = ', algopy_jacobian)

    # reverse mode using a computational graph
    # Step 1/2 - trace the evaluation function
    cg = algopy.CGraph()
    x = algopy.Function([1, 2, 3])
    y = d_f(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]

    # Step 2/2 - use the graph to evaluate derivatives
    print('gradient =', cg.gradient([3., 5, 7]))
    print('Jacobian =', cg.jacobian([3., 5, 7]))  # a square matrix of first order partial derivatives, the derivative of f at all possible points wrt x
    print('Hessian =', cg.hessian([3., 5., 7.]))  # a matrix of second order partial derivatives of the function in question (square), can use optimisation for local min/max/saddle of a critical value.
    print('Hessian vector product =', cg.hess_vec([3., 5., 7.], [4, 5, 6]))

#
# Use-cases of algopy in maths
#

# Posterior Log Probability - the derivative of a posterior log probability calculation w/ a normal dist and known variance
# stemming from Bayes law the posterior probability is the of event A occurring if event B has already occurred in terms of joint Prob theory and the like - could be useful for financial forecasting but not so much on price data which isnt normally distrubted ever.
# this code will also print out the computational grapgh using CGrapgh


def logNormLikelyhood(x, mu, sigma):
    return sigma * (x - mu)**2 - numpy.log(0.5 * sigma / numpy.pi)


def logp(x, mu, sigma):
    return numpy.sum(logNormLikelyhood(x, mu, sigma)) + logNormLikelyhood(mu, mu_prior_mean, mu_prior_sigma)


mu_prior_mean = 1
mu_prior_sigma = 5

actual_mu = 3.1
sigma = 1.2
N = 35
x = numpy.random.normal(actual_mu, sigma, size=N)
mu = UTPM([[3.5], [1], [0]])  # unknown variable

# forward AD
utp = logp(x, mu, sigma).data[:, 0]
print(utp[0], 1. * utp[1], 2. * utp[2])  # prints function evaluation, 1st directional derivative and 2nd directional derivative

print((logp(x, 3.5 + 10**-8, sigma) - logp(x, 3.5, sigma)) / 10**-8)  # prints finite differences derivative

# create trace function and evaluate, saving to a file
cg = CGraph()
mu = Function(UTPM([[3.5], [1], [0]]))  # unknown variable
out = logp(x, mu, sigma)
cg.trace_off()
cg.independentFunctionList = [mu]
cg.dependentFunctionList = [out]
cg.plot(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'posterior_log_probability_cgraph.png'))

# reverse AD
outbar = UTPM([[1.], [0], [0]])
cg.pullback([outbar])
gradient = mu.xbar.data[0, 0]
hessian_vector = mu.data[1, 0]

print(gradient)
print(hessian_vector)

# there are a variety of other uses for algopy but they have miniscule or no apparent gain over existing methods
