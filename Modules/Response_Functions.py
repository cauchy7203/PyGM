import numpy as np


class GM_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, x_0):
        a = params[0]
        b = params[1]
        x1_pred = (x_0 - b / a) * np.exp(-a * t) + b / a
        return x1_pred


class NGM_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, x_0):
        a = params[0]
        b = params[1]
        x1_pred = (x_0 - b / a + b / a / a) * np.exp(-a * t) + b * (t + 1) / a - b / a / a
        return x1_pred


class BernoulliGM_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, x_0, N):
        a = params[0]
        b = params[1]
        x1_pred = ((x_0 ** (1 - N) - b / a) * np.exp(-a * (1 - N) * t) + b / a) ** (1 / (1 - N))
        return x1_pred


class DGM_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, x_0):
        a = params[0]
        b = params[1]
        x1_pred = (a ** t) * (x_0 - (b / (1 - a))) + b / (1 - a)
        return x1_pred


class NDGM_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, x_0):
        b1 = params[0]
        b2 = params[1]
        b3 = params[2]
        n = max(t)
        b4 = 0
        c = 0
        for k in range(1, n):
            for j in range(1, k + 1):
                c_n = j * (b1 ** (k - 1))
                c = c + c_n
            b4_empty = (x_0[k] - (b1 ** k) * x_0[0] - b2 * c - (1 - (b1 ** k)) / (1 - b1) * b3) * (b1 ** k)
            b4 = b4 + b4_empty
        x1_1 = x_0[0] + b4
        x1_pred = np.zeros(n+1)
        x1_pred[0] = x1_1
        for i in range(1, n+1):
            x1_pred[i] = b1 * x1_pred[i - 1] + b2 * i + b3
        return x1_pred


class GMN_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, x1):
        a = params[0]
        b = params[1:]
        lens_0 = len(np.array(x1[0:, 0]))
        c = np.zeros(lens_0)
        for i in range(0, lens_0):
            c[i] = np.sum(b * x1[i, 1:])
        x1_pred = (x1[0][0] - c / a) / np.exp(a * t) + c / a
        return x1_pred


class DGMN_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, x1):
        a = params[0]
        u = params[-1]
        b = params[1:-1]
        lens_0 = len(np.array(x1[0:, 0]))
        c = np.zeros(lens_0)
        for i in range(0, lens_0):
            c[i] = np.sum((a ** i) * np.sum(b * x1[i, 1:]))
        x1_pred = (a ** t) * x1[0][0] + c + (1 - (a ** t) / (1 - a)) * u
        return x1_pred
