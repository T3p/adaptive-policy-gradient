import numpy as np
import math
import os

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
