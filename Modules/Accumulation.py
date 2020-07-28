import numpy as np

def ago(x_ori, p):
    if (p == True):
        x_ori = np.array(list(x_ori))
        x1 = x_ori.cumsum(axis=0)
        return (x1)
    else:
        x_ori = np.array(list(x_ori))
        x0_1 = x_ori[0]
        x0_1 = np.array([x0_1])
        x1 = np.diff(x_ori)
        x1_all = np.concatenate((x0_1, x1), axis=0)
        return (x1_all)


def agom(x_ori, p):
    if (p == True):
        x_ori = np.array(list(x_ori))
        x1 = x_ori.cumsum(axis=0)
        return (x1)
    else:
        x_ori = np.array(list(x_ori))
        x0_1 = x_ori[0]
        x0_1 = np.array([x0_1])
        x1 = np.diff(x_ori,axis=0)
        x1_all = np.concatenate((x0_1, x1), axis=0)
        return (x1_all)