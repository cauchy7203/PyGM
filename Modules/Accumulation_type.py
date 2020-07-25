import numpy as np
class Modelmethod():
    def __init__(self,x_ori,p=True):
        self.x_ori = x_ori
        self.p = p
        self.x1 = None
    def ago(self,x_ori,p=True):
        if (self.p == True):
            self.x1 = self.x.cumsum()
            return (x1)
        else:
            x1 = np.diff(self.x_ori)
            return (x1)

    def basegm(self,x1):
        x1 = np.array(list(self.x1))
        z1 = 0.5*(x1[1:]+x1[:-1])
        z1 = z1.reshape([-1,1])
        return (z1)

    def construct_matrix(self,x_ori,z1):
        lens = len(x_ori)
        ones_array = np.ones([lens-1, 1]).astype(np.float64)
        B = np.concatenate((-z1, ones_array), axis=1)


    def params(self,B,Y):
        Y = Y.reshape([-1,1])
        B_y = np.matmul(B.T, Y[1:, 0]).reshape([-1, 1])
        params = np.matmul(np.linalg.pinv(B), Y[1:, 0])
        return (params)

