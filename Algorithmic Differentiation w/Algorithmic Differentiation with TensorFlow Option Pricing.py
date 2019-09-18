import numpy as np
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as mpl
from matplotlib import cm

# Plain Vanilla BSM pricing


def bsm(S, K, T, r, q, v, option='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * v ** 2) * T) / (v * np.sqrt(T))

    if option == 'call':
        bsm = S * Phi(d1) - K * np.exp(r - q * T) * Phi(d2)
    if option == 'put':
        bsm = K * np.exp(r - q * T) * Phi(d2) - S * Phi(d1)

    return bsm


# Similar technique but using TensorFlow CALL only
# Exterior function sets up static tf graph, using tf formalities
# interior function is fed the params
def bsmtf(enable_greeks=True):
    S = tf.placeholder(tf.float32)
    K = tf.placeholder(tf.float32)
    T = tf.placeholder(tf.float32)
    r = tf.placeholder(tf.float32)
    q = tf.placeholder(tf.float32)
    v = tf.placeholder(tf.float32)
    Phi = tf.distributions.Normal(0., 1.).cdf
    d1 = (tf.log(S / K) + (r + v ** 2 / 2) * T) / (v * tf.sqrt(T))
    d2 = d1 - v * tf.sqrt(T)
    bsm = S * Phi(d1) - K * tf.exp(r - q * T) * Phi(d2)
    target_calc = [bsm]
    if enable_greeks:
        greeks = tf.gradients(bsm, [S, v, r, K, T])
        dS_2ndOrder = tf.gradients(greeks[0], [S, K, T, r, v])  # 2nd order greeks
        dv_2ndOrder = tf.gradients(greeks[1], [S, K, T, r, v])
        dr_2ndOrder = tf.gradients(greeks[2], [S, K, T, r, v])
        dK_2ndOrder = tf.gradients(greeks[3], [S, K, T, r, v])
        dT_2ndOrder = tf.gradients(greeks[4], [S, K, T, r, v])
        target_calc += [greeks, dS_2ndOrder, dv_2ndOrder, dr_2ndOrder, dK_2ndOrder, dT_2ndOrder]

    def graph_execute(S, K, T, r, q, v):
        with tf.Session() as sess:
            res = sess.run(target_calc,
                           {
                               S: S,
                               K: strike,
                               r: riskfree_rate,
                               v: implied_vol,
                               T: time_to_expiry
                               q: dividend})
            return res
        return graph_execute


# tf_pricer = bsmtf()
# tf_pricer(..., True/False)
