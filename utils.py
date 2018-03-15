import numpy as np
import math
import os

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

def zero_fun(x):
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


def calc_sigma(K, Q, R, gamma):
    P = calc_P(K, Q, R, gamma)
    return -(R + gamma*P)/(1 - gamma)


def calc_mixed(gamma, theta, R, Q):
    den = 1 - gamma*(1 + 2*theta + theta**2)
    dePdeK = 2*(theta*R/den + gamma*(Q + theta**2*R)*(1+theta)/den**2)

    return -dePdeK*gamma/(1 - gamma)



#
#   COMPILE IF NUMBA IS PRESENT
#

if NUMBA_PRESENT:
    calc_K = numba.jit(calc_K, nopython=True)
    calc_P = numba.jit(calc_P, nopython=True)
    calc_J = numba.jit(calc_J, nopython=True)
    calc_sigma = numba.jit(calc_sigma, nopython=True)
    calc_mixed = numba.jit(calc_mixed, nopython=True)
