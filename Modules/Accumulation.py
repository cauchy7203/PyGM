import numpy as np


def ago(x_ori, x0, p):
    if (p == True):
        x_ori = np.array(list(x_ori))
        x1 = x_ori.cumsum(axis=0)
        return (x1)
    else:
        x_ori = np.array(list(x_ori))
        x1 = np.diff(x_ori)
        x1_all = np.concatenate(([x0], x1), axis=0)
        return (x1_all)


def agom(x_ori, x0, p):
    if (p == True):
        x_ori = np.array(list(x_ori))
        x1 = x_ori.cumsum(axis=0)
        return (x1)
    else:
        x_ori = np.array(list(x_ori))
        x1 = np.diff(x_ori, axis=0)
        x1_all = np.concatenate(([x0], x1), axis=0)
        return (x1_all)
