from CommonOperations import ModelMethods
from CommonOperations import _res_func_list
from CommonOperations import Accumulation
import numpy as np


class GM():
    def __init__(self):
        self.mdl_name = 'gm11'
        self.is_fitted = False

    def fit(self, x, y):
        x1 = Accumulation.ago(y, None, True)
        Y = x1[1:] - x1[0: -1]
        z1 = ModelMethods.get_backvalue(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        self.B = ModelMethods.construct_matrix(z1, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(self.B, Y)
        self.is_fitted = True
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = Accumulation.ago(x1_pred, self.x_orig[0], False)
        return x_pred


class NGM():
    def __init__(self):
        self.mdl_name = 'ngm'
        self.is_fitted = False

    def fit(self, x, y):
        x1 = Accumulation.ago(y, None, True)
        Y = x1[1:] - x1[0: -1]
        z1 = ModelMethods.get_backvalue(x1)
        ones_array = np.diff(x, axis=0).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        arange_array = x[1:]
        arange_array = arange_array.reshape([-1, 1])
        B1 = ModelMethods.construct_matrix(z1, arange_array)
        self.B = ModelMethods.construct_matrix(-B1, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(self.B, Y)
        self.is_fitted = True
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = Accumulation.ago(x1_pred, self.x_orig[0], False)
        return x_pred


class BernoulliGM():
    def __init__(self, n=2):
        self.mdl_name = 'bergm'
        self.n = n
        self.is_fitted = False

    def fit(self, y):
        x1 = Accumulation.ago(y, None, True)
        Y = x1[1:] - x1[0: -1]
        z1 = ModelMethods.get_backvalue(x1)
        z1_square = np.power(z1, self.n)
        self.B = ModelMethods.construct_matrix(z1, z1_square)
        self.x_orig = y
        self.params = ModelMethods.get_params(self.B, Y)
        self.is_fitted = True
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0], self.n)
        x_pred = Accumulation.ago(x1_pred, self.x_orig[0], False)
        return x_pred


class DGM():
    def __init__(self):
        self.mdl_name = 'dgm11'
        self.is_fitted = False

    def fit(self, x, y):
        x1 = Accumulation.ago(y, None, True)
        z1 = ModelMethods.based(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        self.B = ModelMethods.construct_matrix(-z1, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(self.B, x1[1:])
        self.is_fitted = True
        return self

    def predict(self, t):
        all_t = np.arange(0, np.max(t))
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = Accumulation.ago(x1_pred, self.x_orig[0], False)
        return x_pred


class NDGM():
    def __init__(self):
        self.mdl_name = 'ndgm'
        self.is_fitted = False

    def fit(self, x, y):
        x1 = Accumulation.ago(y, None, True)
        z1 = ModelMethods.based(x1)
        ones_array = np.diff(x).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        range_array = np.arange(len(x) - 1)+1
        range_array = range_array.reshape([-1, 1])
        B1 = ModelMethods.construct_matrix(-z1, range_array)
        self.B = ModelMethods.construct_matrix(-B1, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(self.B, x1[1:])
        self.x1 = x1
        self.is_fitted = True
        return self

    def predict(self, t):
        all_t = np.arange(1, np.max(t)+1)
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x1)
        self.x_pred = Accumulation.ago(x1_pred, x1_pred[0], False)
        x_pred = self.x_pred[-len(t):]
        return x_pred

class GMN():
    def __init__(self):
        self.mdl_name = 'gm1n'
        self.is_fitted = False

    def fit(self, y, x):
        x_train = x[0:, 1:]
        y_conect = np.concatenate((y, x_train), axis=1)
        x1 = Accumulation.agom(y_conect, None, True)
        x1_0 = x1[0:, 0]
        z1 = ModelMethods.get_backvalue(x1_0)
        n_array = x1[1:, 1:]
        self.B = ModelMethods.construct_matrix(z1, n_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(self.B, np.array(y_conect)[1:, 0])
        self.x1 = x1
        self.is_fitted = True
        return self

    def predict(self, x):
        all_t = x[0:, 0]
        x_test = x[0:, 1:]
        x1_test = Accumulation.agom(x_test, None,True)
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0][0], x1_test)
        x_pred = Accumulation.agom(x1_pred, self.x_orig[0][0], False)
        return x_pred


class DGMN():
    def __init__(self):
        self.mdl_name = 'dgm1n'
        self.is_fitted = False

    def fit(self, y, x):
        x_train = x[0:, 1:]
        t = x[0:,0]
        y_conect = np.concatenate((y, x_train), axis=1)
        x1 = Accumulation.agom(y_conect, None, True)

        ones_array = np.diff(t).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        x1_0 = x1[0:-1, 0]
        x1_0 = x1_0.reshape([-1, 1])
        x1_n = x1[1:, 1:]
        B_x = ModelMethods.construct_matrix(-x1_0, x1_n)
        self.B = ModelMethods.construct_matrix(-B_x, ones_array)
        self.x_orig = y
        self.params = ModelMethods.get_params(self.B, x1[1:,0])
        self.x1 = x1
        self.is_fitted = True
        return self

    def predict(self, t):
        all_t = t[0:, 0]
        x_test = t[0:, 1:]
        x1_test = Accumulation.agom(x_test, None,True)
        x1_pred = _res_func_list.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0][0],x1_test)
        x1_pred = np.concatenate(([self.x_orig[0][0]], x1_pred), axis=0)
        x_pred = Accumulation.agom(x1_pred, self.x_orig[0][0], False)
        return x_pred