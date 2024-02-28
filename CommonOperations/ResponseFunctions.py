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
        c = params[2]
        x1_pred = (x_0 - b / a + b / a / a - c / a) * np.exp(-a * t) + b * (t + 1) / a - b / a / a + c / a
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

    def compute(self, params, t, x1):
        b1 = params[0]
        b2 = params[1]
        b3 = params[2]
        n = len(x1)
        E1 = 0
        E2 = 0
        for i in range(1, n):
            E3 = 0
            for j in range(1, i + 1):
                E3 += j * np.power(b1, i - j)
            e1 = ((1 - np.power(b1, i)) / (1 - b1)) * b3
            E1 += (x1[i] - np.power(b1, i) * x1[0] - b2 * E3 - e1) * np.power(b1, i)
            E2 += np.power(b1, 2 * i)
            b4 = E1 / (1 + E2)

        x1_pred = np.zeros(max(t))
        x1_pred[0] = x1[0] + b4
        for i in range(1, max(t)):
            E4 = 0
            for j in range(1, i + 1):
                E4 += j * (b1 ** (i - j))
            x1_pred[i] = (b1 ** i) * x1_pred[0] + b2 * E4 + ((1 - b1 ** i) / (1 - b1)) * b3
        return x1_pred


class GMN_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, y_0, x1):

        a = params[0]
        b = params[1:].T
        c = np.dot(b, x1[:, :].T)[0]
        x1_pred = (y_0 - c[:] / a) / np.exp(a * (t[:] - 1)) + c[:] / a
        return x1_pred


class DGMN_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, t, y_0, x1):

        t_use = t[:-1]
        a = params[0]
        u = params[-1]
        b = params[1:-1]
        lens_0 = len(np.array(x1[:-1, 0]))
        c = np.zeros(lens_0)
        temp1 = np.dot(x1[1:, :], b)
        temp2 = 0
        for i in range(0, lens_0):
            temp2 = temp2 + temp1[i]
            c[i] = temp2
            temp2 = temp2 * a
        x1_pred = (a ** t_use) * y_0 + c + ((1 - (a ** t_use)) / (1 - a)) * u
        return x1_pred

