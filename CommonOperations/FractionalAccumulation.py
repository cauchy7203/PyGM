from scipy.special import binom
import numpy as np
from numpy import cumprod
from math import factorial as fact

# compute the r-order accumulation
def fago(x,r):
    n = len(x)
    i = n - np.arange(n)
    fnum = n - i + r -1
    fnum[0] = 1
    fnum_prod = np.flip(cumprod(fnum))

    dom = np.array( [fact(x) for x in i-1])

    # coefs in the last column of the matrix
    coefs = fnum_prod / dom
    xr = np.zeros(n)
    for k in range(n):
        xr[k] = sum(x[:k+1]*coefs[-(k+1):])

    return xr