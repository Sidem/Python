from time import time
import numpy as np

LEAKY_RELU = True

def timeit(func):
    def wrapper(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print('time elapsed: ', after - before, 'seconds')
        return rv
    return wrapper

def relu_derivative(x):
    if x > 0:
        return 1
    else:
        if LEAKY_RELU:
            return 0.01
        else:
            return 0

def relu(x, derivative=False):
    """
    arguments:    
        x:          numpy array
        derivative: boolean
    return:
        rectification of x ( x if x is positive else 0 or x/100(leaky) )
        if derivative is true -> 1 or 0/0.01(leaky)
    """
    if derivative:
        return np.fromiter((relu_derivative(xi) for xi in x), x.dtype)
    else:
        if LEAKY_RELU:
            return np.maximum(x, x*0.01, x)
        else:
            return np.maximum(x, 0, x)

def get_error(result, target):
    return np.sum(np.power(result - target, 2))

def get_delta(result, target):
    return result - target