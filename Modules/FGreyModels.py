from CommonOperations import ModelMethods
from CommonOperations import _res_func_list
from CommonOperations import FractionalAccumulation
import numpy as np


class FGM():
    def __init__(self, r=0.5):
        self.mdl_name = 'gm11'
        self.r = r
        self.is_fitted = False

    def fit(self, x, y):
        x1 = FractionalAccumulation.fago(y, self.r)
        Y = x1[1:] - x1[0: -1]
        z1 = ModelMethods.get_backvalue(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        B = ModelMethods.construct_matrix(z1, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(B, Y)
        self.is_fitted = True
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = FractionalAccumulation.fago(x1_pred, -self.r)
        return x_pred


class FNGM():
    def __init__(self, r=0.5):
        self.mdl_name = 'ngm'
        self.r = r

    def fit(self, x, y):
        x1 = FractionalAccumulation.fago(y, self.r)
        Y = x1[1:] - x1[0: -1]
        z1 = ModelMethods.get_backvalue(x1)
        ones_array = np.diff(x, axis=0).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        arange_array = x[1:]
        arange_array = arange_array.reshape([-1, 1])
        B = ModelMethods.construct_matrix(z1, arange_array)
        B = ModelMethods.construct_matrix(-B, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(B, Y)
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = FractionalAccumulation.fago(x1_pred, -self.r)
        return x_pred


class FBernoulliGM():
    def __init__(self, n=2, r=0.5):
        self.mdl_name = 'bergm'
        self.n = n
        self.r = r

    def fit(self, y):
        x1 = FractionalAccumulation.fago(y, self.r)
        Y = x1[1:] - x1[0: -1]
        z1 = ModelMethods.get_backvalue(x1)
        z1_square = np.power(z1, self.n)
        B = ModelMethods.construct_matrix(z1, z1_square)
        self.x_orig = y
        self.params = ModelMethods.get_params(B, Y)
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0], self.n)
        x_pred = FractionalAccumulation.fago(x1_pred, -self.r)
        return x_pred


class FDGM():
    def __init__(self, r=0.5):
        self.mdl_name = 'dgm11'
        self.r = r

    def fit(self, x, y):
        x1 = FractionalAccumulation.fago(y, self.r)
        z1 = ModelMethods.based(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        B = ModelMethods.construct_matrix(-z1, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(B, x1[1:])
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = FractionalAccumulation.fago(x1_pred, -self.r)
        return x_pred


class FNDGM():
    def __init__(self, r=0.5):
        self.mdl_name = 'ndgm'
        self.r = r

    def fit(self, x, y):
        x1 = FractionalAccumulation.fago(y, self.r)
        z1 = ModelMethods.based(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        range_array = np.arange(len(x) - 1)+1
        range_array = range_array.reshape([-1, 1])
        B1 = ModelMethods.construct_matrix(-z1, range_array)
        B = ModelMethods.construct_matrix(-B1, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(B, x1[1:])
        self.x1 = x1
        return self

    def predict(self, t):
        all_t = np.arange(1, np.max(t)+1)
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x1)
        x_pred = FractionalAccumulation.fago(x1_pred, -self.r)
        return x_pred


class FGMN():
    def __init__(self, r=0.5):
        self.mdl_name = 'gm1n'
        self.r = r

    def fit(self, y, x):
        x_train = x[0:, 1:]
        y_conect = np.concatenate((y, x_train), axis=1)
        lens_1 = len(y_conect[0, 0:])
        y1 = y_conect.T
        x1 = np.zeros(y1.shape)
        for i in range(0, lens_1):
            x1[i, 0:] = FractionalAccumulation.fago(y1[i, 0:], self.r)
        x1 = x1.T
        x1_0 = x1[0:, 0]
        z1 = ModelMethods.get_backvalue(x1_0)
        n_array = x1[1:, 1:]
        B = ModelMethods.construct_matrix(z1, n_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(B, np.array(y)[1:, 0])
        self.x1 = x1
        return self

    def predict(self, x):
        all_t = x[0:, 0]
        x_test = x[0:, 1:]
        x_test = x_test.T
        x1_test = np.zeros(x_test.shape)
        for j in range(0, len(x_test[:, 0])):
            x1_test[j, 0:] = FractionalAccumulation.fago(x_test[j, 0:], self.r)
        x1_test = x1_test.T
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0][0], x1_test)
        x_pred = FractionalAccumulation.fago(x1_pred, -self.r)
        return x_pred

class FDGMN():
    def __init__(self, r=0.5):
        self.mdl_name = 'dgm1n'
        self.r = r

    def fit(self, y, x):
        x_train = x[0:, 1:]
        t = x[0:, 0]
        y_conect = np.concatenate((y, x_train), axis=1)
        lens_1 = len(y_conect[0, 0:])
        y1 = y_conect.T
        x1 = np.zeros(y1.shape)
        for i in range(0, lens_1):
            x1[i, 0:] = FractionalAccumulation.fago(y1[i, 0:], self.r)
        x1 = x1.T
        ones_array = np.diff(t).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        x1_0 = x1[0:-1, 0]
        x1_0 = x1_0.reshape([-1, 1])
        x1_n = x1[1:, 1:]
        B_x = ModelMethods.construct_matrix(-x1_0, x1_n)
        B = ModelMethods.construct_matrix(-B_x, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(B, x1[1:, 0])
        self.x1 = x1
        return self

    def predict(self, x):
        all_t = x[0:, 0]
        x_test = x[0:, 1:]
        x_test = x_test.T
        x1_test = np.zeros(x_test.shape)
        for j in range(0, len(x_test[:, 0])):
            x1_test[j, 0:] = FractionalAccumulation.fago(x_test[j, 0:], self.r)
        x1_test = x1_test.T
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0][0], x1_test)
        x1_pred = np.concatenate(([self.x_orig[0][0]], x1_pred), axis=0)
        x_pred = FractionalAccumulation.fago(x1_pred, -self.r)


        return x_pred
