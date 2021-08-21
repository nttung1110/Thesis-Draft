import numpy as np

from re import L


def sigmoid_func(value):
    return 1/(1+np.exp(-value))