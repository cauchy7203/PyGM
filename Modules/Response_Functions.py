import numpy as np



class GM_ResFunc():
    def __init__(self):
        pass

    def compute(self, params, data_pre, x_orig):
        a = params[0]
        b = params[1]
        lens = len(x_orig)
        k_prediction = np.arange(0, data_pre + lens)
        x1_pre = (x_orig[0] - b / a) * np.exp(-a * k_prediction) + b / a
        return (x1_pre)

    # def ngm(self, params, data_pre, x_orig):
    #     a = params[0]
    #     b = params[1]
    #     lens = len(x_orig)
    #     k_prediction = np.arange(0, data_pre + lens)
    #     x1_pre = (x_orig[0] - b / a + b / a / a) * np.exp(-a * k_prediction) + b * k_prediction / a - b / a / a
    #     return (x1_pre)
    #
    # def vergm(self, params, data_pre, x_orig):
    #     a = params[0]
    #     b = params[1]
    #     lens = len(x_orig)
    #     k_prediction = np.arange(0, data_pre + lens)
    #     x1_pre = (a * x_orig[0]) / (b * x_orig[0] + (a - b * x_orig[0]) * np.exp(a * k_prediction))
    #     return (x1_pre)
