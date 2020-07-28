from CommonOperation import Response_Functions
from CommonOperation import ModelMethod
from CommonOperation import _res_funcs
import numpy as np


class GM():
    def __init__(self):
        self.mdl_name = 'gm11'

    def fit(self, t, x_orig):
        x1 = ModelMethod.ago(x_orig, True)
        z1 = ModelMethod.basegm(x1)
        ones_array = np.diff(t).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        B = ModelMethod.construct_matrix(z1, ones_array)
        self.x_orig = x_orig
        self.params = ModelMethod.params(B, x_orig)
        return self

    def predict(self, t):
        all_t = np.arange(t[-1])
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = ModelMethod.ago(x1_pred, False)
        return x_pred


class NGM():
    def __init__(self):
        self.mdl_name = 'ngm11'

    def fit(self, t, x_orig):
        x1 = ModelMethod.ago(x_orig, True)
        z1 = ModelMethod.basegm(x1)
        arange_array = t[1:]
        arange_array = arange_array.reshape([-1, 1])
        B = ModelMethod.construct_matrix(z1, arange_array)
        self.x_orig = x_orig
        self.params = ModelMethod.params(B, x_orig)
        return self

    def predict(self, t):
        all_t = np.arange(t[-1])
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = ModelMethod.ago(x1_pred, False)
        return x_pred


class BernoulliGM():
    def __init__(self):
        self.mdl_name = 'bergm'

    def fit(self, N, x_orig):
        x1 = ModelMethod.ago(x_orig, True)
        z1 = ModelMethod.basegm(x1)
        z1_square = np.power(z1, N)
        B = ModelMethod.construct_matrix(z1, z1_square)
        self.x_orig = x_orig
        self.params = ModelMethod.params(B, x_orig)
        self.N = N
        return self

    def predict(self, t):
        all_t = np.arange(t[-1])
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0], self.N)
        x_pred = ModelMethod.ago(x1_pred, False)
        return x_pred


class DGM():
    def __init__(self):
        self.mdl_name = 'dgm11'

    def fit(self, t, x_orig):
        x1 = ModelMethod.ago(x_orig, True)
        z1 = ModelMethod.based(x1)
        ones_array = np.diff(t).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        B = ModelMethod.construct_matrix(-z1, ones_array)
        self.x_orig = x_orig
        self.params = ModelMethod.params(B, x1)
        return self

    def predict(self, t):
        all_t = np.arange(t[-1])
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x_orig[0])
        x_pred = ModelMethod.ago(x1_pred, False)
        return x_pred


class GMN():
    def __init__(self):
        self.mdl_name = 'gm1n'

    def fit(self, t, x_orig):
        x1 = ModelMethod.agom(x_orig, True)
        x1_0 = x1[0:,0]
        z1 = ModelMethod.basegm(x1_0)
        n_array = x1[1:,1:]
        B = ModelMethod.construct_matrix(z1, n_array)
        self.x_orig = x_orig
        self.params = ModelMethod.params(B, np.array(x_orig)[0:,0])
        self.x1 = x1
        return self

    def predict(self, t):
        all_t = np.arange(t[-1])
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x1)
        x_pred = ModelMethod.agom(x1_pred, False)
        return x_pred


class DGMN():
    def __init__(self):
        self.mdl_name = 'dgm1n'

    def fit(self, t, x_orig):
        x1 = ModelMethod.agom(x_orig, True)
        ones_array = np.diff(t).astype(np.float64)
        ones_array = ones_array.reshape([-1, 1])
        x1_0 = x1[0:,0]
        x1_n = x1[1:,1:]
        B_x = ModelMethod.construct_matrix(-x1_0, x1_n)
        B = ModelMethod.construct_matrix(-B_x, ones_array)
        self.x_orig = x_orig
        self.params = ModelMethod.params(B, np.array(x_orig)[0:, 0])
        self.x1 = x1

    def predict(self, t):
        all_t = np.arange(t[-1])
        x1_pred = _res_funcs.res_funcs[self.mdl_name].compute(self.params, all_t, self.x1)
        x_pred = ModelMethod.agom(x1_pred, False)
        return x_pred



# test
# g = GM('gm11')
# g.fit([15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5])
# g.predict(5)
#
# n = NGM('ngm11')
# n.fit([15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5])
# n.predict(5)
#
# v = VerhulstGM('vergm')
# v.fit([15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5])
# v.predict(5)
#
# d = DGM('dgm11')
# d.fit([15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5])
# d.predict(5)
