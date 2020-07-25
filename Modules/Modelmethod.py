import numpy as np
def ago(x_ori, p=True):
    if (p == True):
        x_ori = np.array(list(x_ori))
        x1 = x_ori.cumsum()
        return (x1)
    else:
        x_ori = np.array(list(x_ori))
        x0_1 = x_ori[0]
        x0_1 = np.array([x0_1])
        x1 = np.diff(x_ori)
        x1_all = np.concatenate((x0_1, x1), axis=0)
        return (x1_all)


def basegm(x1):
    x1 = np.array(list(x1))
    z1 = 0.5 * (x1[1:] + x1[:-1])
    z1 = z1.reshape([-1, 1])
    return (z1)


def construct_matrix(z1,ones_array):
    B = np.concatenate((-z1, ones_array), axis=1)
    return (B)


def params(B, Y):
    Y = np.array(list(Y))
    Y = Y.reshape([-1, 1])
    B_y = np.matmul(B.T, Y[1:, 0]).reshape([-1, 1])
    params = np.matmul(np.linalg.pinv(B), Y[1:, 0])
    return (params)