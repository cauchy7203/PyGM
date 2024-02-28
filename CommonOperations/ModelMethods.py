import numpy as np


def get_backvalue(x1):
    x1 = np.array(list(x1))
    z1 = 0.5 * (x1[1:] + x1[:-1])
    z1 = z1.reshape([-1, 1])
    return (z1)


def based(x1):
    x1 = np.array(list(x1))
    z1 = x1[0:-1]
    z1 = z1.reshape([-1, 1])
    return (z1)


def construct_matrix(z1, ones_array):
    B = np.concatenate((-z1, ones_array), axis=1)
    return (B)


def get_params(B, Y):
    Y = np.array(list(Y))
    Y = Y.reshape([-1, 1])
    params = np.matmul(np.linalg.pinv(B), Y)
    return (params)
