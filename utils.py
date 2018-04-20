import numpy as np
import math
import os
import random
import string

try:
    import numba
    NUMBA_PRESENT = True
except ImportError:
    NUMBA_PRESENT = False

"""Helper functions"""


def apply_along_axis2(func1d,axis,X,Y):
    """Extension of numpy.apply_along_axis to functions of two parameters"""
    if len(X.shape)<=axis:
        X = np.expand_dims(X,axis=axis)

    if len(Y.shape)<=axis:
        Y = np.expand_dims(Y,axis=axis)

    split = X.shape[axis]
    Z = np.concatenate((X,Y),axis)

    def aux(z):
        return func1d(z[:split],z[split:])

    return np.apply_along_axis(aux,axis,Z)

def identity(x):
    """Identity function"""
    return x

def zero_fun(x, deterministic=None):
    """Null function"""
    return 0

def maybe_make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def range_unlimited(r=-1):
    n = 0
    while n != r:
        yield n
        n += 1

def generate_filename():
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))

def split_batch_sizes(batch_size, n):
    min_num = min(batch_size, 1)
    x = batch_size
    r = [0] * n
    i = 0
    max_num = max(math.ceil(batch_size / n), min_num)
    while x > 0:
        r[i] = min(max_num, x)
        x -= r[i]
        i+=1

    return r

def is_diagonal(x):
    return np.count_nonzero(x - np.diag(np.diagonal(x))) == 0

#
#   LQG SCALAR SPECIFIC FUNCTIONS
#



def calc_K(theta, sigma, gamma, R, Q, max_pos):
    den = 1 - gamma*(1 + 2*theta + theta**2)
    dePdeK = 2*(theta*R/den + gamma*(Q + theta**2*R)*(1+theta)/den**2)
    return - dePdeK*(max_pos**2/3 + gamma*sigma/(1 - gamma))

def calc_P(K, Q, R, gamma):
    P = (Q + K* R * K) / (1 - gamma * (1 + 2 * K + K**2))
    return P


def calc_J(K, Q, R, gamma, sigma, max_pos, B):
    P = calc_P(K, Q, R, gamma)
    W =  (1 / (1 - gamma)) * sigma * (R + gamma * B*P*B)

    return min(0,-max_pos**2*P/3 - W)

def calc_optimal_sigma(K, Q, R, gamma, B, exp_cost):
    P = calc_P(K, Q, R, gamma)
    return (exp_cost * (1 - gamma)) / (R + gamma*B*P*B)

def calc_sigma(K, Q, R, gamma):
    P = calc_P(K, Q, R, gamma)
    return -(R + gamma*P)/(1 - gamma)


def calc_mixed(gamma, theta, R, Q):
    den = 1 - gamma*(1 + 2*theta + theta**2)
    dePdeK = 2*(theta*R/den + gamma*(Q + theta**2*R)*(1+theta)/den**2)

    return -dePdeK*gamma/(1 - gamma)

def computeLoss(R, M, gamma, volume, sigma):
    return float(R*M**2)/((1-gamma)**2*sigma**2)* \
        (float(volume)/math.sqrt(2*math.pi*sigma**2) + \
            float(gamma)/(2*(1-gamma)))


def computeLossSigma_old(R, M, gamma, volume, sigma):
    c = (4*(math.sqrt(7) - 2)*math.exp((math.sqrt(7))/(2) - 2)) / (math.sqrt(2*math.pi))
    return R/((1-gamma)**2 *sigma) * ((c*volume) / (2) + (gamma) / ((1-gamma)*sigma))

def computeLossSigma(R, M, gamma, volume, sigma):
    c = (4*(math.sqrt(7) - 2)*math.exp((math.sqrt(7))/(2) - 2)) / (math.sqrt(2*math.pi))
    return R/((1-gamma)**2) * ((c*volume) / (2*sigma) + (gamma) / ((1-gamma)))

#
#   COMPILE IF NUMBA IS PRESENT
#

if NUMBA_PRESENT:
    calc_K = numba.jit(calc_K)
    calc_P = numba.jit(calc_P)
    calc_J = numba.jit(calc_J)
    calc_sigma = numba.jit(calc_sigma)
    calc_mixed = numba.jit(calc_mixed)
    computeLoss = numba.jit(computeLoss)
    computeLossSigma = numba.jit(computeLossSigma)
    calc_optimal_sigma = numba.jit(calc_optimal_sigma)
