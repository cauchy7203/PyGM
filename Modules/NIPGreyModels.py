from CommonOperation import ModelMethod
from CommonOperation import _res_funcs
from CommonOperation import NewInformationPriorityAccumulation
import numpy as np


class NIPGM():
    def __init__(self, r=0.99):
        self.mdl_name = 'gm11'
        self.r = r

    def fit(self, x, y):
        x1 = NewInformationPriorityAccumulation.nipago(y, self.r)
        z1 = ModelMethod.get_backvalue(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        B = ModelMethod.construct_matrix(z1, ones_array)
        self.x_orig = y
        self.params = ModelMethod.get_params(B, y)
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = NewInformationPriorityAccumulation.nipago(x1_pred, -self.r)
        return x_pred


class NIPNGM():
    def __init__(self, r=0.99):
        self.mdl_name = 'ngm11'
        self.r = r

    def fit(self, x, y):
        x1 = NewInformationPriorityAccumulation.nipago(y, self.r)
        z1 = ModelMethod.get_backvalue(x1)
        arange_array = x[1:]
        arange_array = arange_array.reshape([-1, 1])
        B = ModelMethod.construct_matrix(z1, arange_array)
        self.x_orig = y
        self.params = ModelMethod.get_params(B, y)
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = NewInformationPriorityAccumulation.nipago(x1_pred, -self.r)
        return x_pred


class NIPBernoulliGM():
    def __init__(self, n=2, r=0.99):
        self.mdl_name = 'bergm'
        self.n = n
        self.r = r

    def fit(self, y):
        x1 = NewInformationPriorityAccumulation.nipago(y, self.r)
        z1 = ModelMethod.get_backvalue(x1)
        z1_square = np.power(z1, self.n)
        B = ModelMethod.construct_matrix(z1, z1_square)
        self.x_orig = y
        self.params = ModelMethod.get_params(B, y)
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0], self.n)
        x_pred = NewInformationPriorityAccumulation.nipago(x1_pred, -self.r)
        return x_pred


class NIPDGM():
    def __init__(self, r=0.99):
        self.mdl_name = 'dgm11'
        self.r = r

    def fit(self, x, y):
        x1 = NewInformationPriorityAccumulation.nipago(y, self.r)
        z1 = ModelMethod.based(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        B = ModelMethod.construct_matrix(-z1, ones_array)
        self.x_orig = y
        self.params = ModelMethod.get_params(B, x1)
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = NewInformationPriorityAccumulation.nipago(x1_pred, -self.r)
        return x_pred


class NIPNDGM():
    def __init__(self, r=0.99):
        self.mdl_name = 'ndgm'
        self.r = r

    def fit(self, x, y):
        x1 = NewInformationPriorityAccumulation.nipago(y, self.r)
        z1 = ModelMethod.based(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        range_array = np.arange(len(x) - 1)
        range_array = range_array.reshape([-1, 1])
        B1 = ModelMethod.construct_matrix(-z1, range_array)
        B = ModelMethod.construct_matrix(-B1, ones_array)
        self.x_orig = y
        self.params = ModelMethod.get_params(B, x1)
        self.x1 = x1
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x1)
        x_pred = NewInformationPriorityAccumulation.nipago(x1_pred, -self.r)
        return x_pred


class NIPGMN():
    def __init__(self, r=0.99):
        self.mdl_name = 'gm1n'
        self.r = r

    def fit(self, x, y):
        lens_1 = len(y[0, 0:])
        y1 = y.T
        x1 = np.zeros(y1.shape)
        for i in range(0, lens_1):
            x1[i, 0:] = NewInformationPriorityAccumulation.nipago(y1[i, 0:], self.r)
        x1 = x1.T
        x1_0 = x1[0:, 0]
        z1 = ModelMethod.get_backvalue(x1_0)
        n_array = x1[1:, 1:]
        B = ModelMethod.construct_matrix(z1, n_array)
        self.x_orig = y
        self.params = ModelMethod.get_params(B, np.array(y)[0:, 0])
        self.x1 = x1
        return x1

    def predict(self, t_x):
        all_t = t_x[0:, 0]
        x1 = t_x[0:, 1:]
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, x1)
        x_pred = NewInformationPriorityAccumulation.nipago(x1_pred, -self.r)
        return x_pred


class NIPDGMN():
    def __init__(self, r=0.99):
        self.mdl_name = 'dgm1n'
        self.r = r

    def fit(self, x, y):
        lens_1 = len(y[0, 0:])
        y1 = y.T
        x1 = np.zeros(y1.shape)
        for i in range(0, lens_1):
            x1[i, 0:] = NewInformationPriorityAccumulation.nipago(y1[i, 0:], self.r)
        x1 = x1.T
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        x1_0 = x1[0:-1, 0]
        x1_0 = x1_0.reshape([-1, 1])
        x1_n = x1[1:, 1:]
        B_x = ModelMethod.construct_matrix(-x1_0, x1_n)
        B = ModelMethod.construct_matrix(-B_x, ones_array)
        self.x_orig = y
        self.params = ModelMethod.get_params(B, np.array(y)[0:, 0])
        self.x1 = x1
        return x1

    def predict(self, t_x):
        all_t = t_x[0:, 0]
        x1 = t_x[0:, 1:]
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, x1)
        x_pred = NewInformationPriorityAccumulation.nipago(x1_pred, -self.r)
        return x_pred
