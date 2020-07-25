from M import Response_Functions
from M import Modelmethod
import numpy as np
class Gm():
    def __init__(self):
        x_orig = x_orig
        self.data_pre = data_pre
        self.params = []

    def fit(self, t = None, x_orig):

        if t == None:
            t = np.reshape(np.arange(len(x_orig)),[-1,1])

        x1 = Modelmethod.ago(self.x_orig,True)
        z1 = Modelmethod.basegm(x1)
        lens = len(self.x_orig)
        ones_array = np.ones([lens - 1, 1]).astype(np.float64)
        B = Modelmethod.construct_matrix(z1,ones_array)
        self.params = Modelmethod.params(B,self.x_orig)
        return self

    def predict(self,t):

        x1_pre = Response_Functions.Resfuncs.gm(self,self.params, self.data_pre, self.x_orig)
        x_pre = Modelmethod.ago(x1_pre,False)
        print(x_pre)


class Ngm():
    def __init__(self,x_orig,data_pre):
        self.x_orig = x_orig
        self.data_pre = data_pre
        self.params = []

    def fit(self):

        x1 = Modelmethod.ago(self.x_orig,True)
        z1 = Modelmethod.basegm(x1)
        lens = len(self.x_orig)
        arange_array = np.arange(2, lens + 1).astype(np.float)  # astype目的是化为浮点型
        arange_array = arange_array.reshape([-1, 1])
        B = Modelmethod.construct_matrix(z1,arange_array)
        self.params = Modelmethod.params(B,self.x_orig)

    def prediction(self):

        x1_pre = Response_Functions.Resfuncs.ngm(self, self.params, self.data_pre, self.x_orig)
        x_pre = Modelmethod.ago(x1_pre, False)
        print(x_pre)


class Vergm():
    def __init__(self,x_orig,data_pre):
        self.x_orig = x_orig
        self.data_pre = data_pre
        self.params = []

    def fit(self):
        x1 = Modelmethod.ago(self.x_orig, True)
        z1 = Modelmethod.basegm(x1)
        lens = len(self.x_orig)
        z1_square = np.square(z1)
        B = Modelmethod.construct_matrix(z1, z1_square)
        self.params = Modelmethod.params(B, self.x_orig)

    def prediction(self):

        x1_pre = Response_Functions.Resfuncs.vergm(self, self.params, self.data_pre, self.x_orig)
        x_pre = Modelmethod.ago(x1_pre, False)
        print(x_pre)




#test
g = Gm([15,16.1,17.3,18.4,18.7,19.6,19.9,21.3,22.5],5)
g.fit()
g.predict()

n = Ngm([15,16.1,17.3,18.4,18.7,19.6,19.9,21.3,22.5],5)
n.fit()
n.prediction()

v = Vergm([15,16.1,17.3,18.4,18.7,19.6,19.9,21.3,22.5],5)
v.fit()
v.prediction()

