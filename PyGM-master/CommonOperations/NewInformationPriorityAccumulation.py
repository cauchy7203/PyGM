import numpy as np


def nipago(x, r):
    lens = len(x)
    x1 = np.zeros([lens, lens])
    if (r > 0):
        for i in range(0, lens):
            for j in range(0, lens):
                if i <= j:
                    x1[i, j] = r ** (j - i)
        xr = np.dot(x, x1)
        return xr
    else:
        for i in range(0, lens):
            for j in range(0, lens):
                if ((i < j and (j - i) == 1) or i == j):
                    x1[i, j] = r ** (j - i)
        xr = np.dot(x, x1)
        return xr
