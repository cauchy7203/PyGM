from Modules.NIPGreyModels import NIPGMN
import numpy as np

y = np.array([2.874, 3.278, 3.307, 3.39, 3.679]).reshape(-1, 1)
x = np.array([7.04, 7.645, 8.075, 8.53, 8.774]).reshape(-1, 1)
t = np.arange(1, len(y) + 1).reshape(-1, 1)
t_x = np.concatenate((t, x), axis=1)

nipgmm = NIPGMN()
nipgmm.fit(t_x[:4,:], y[:4,:])

y_predict = nipgmm.predict(t_x[:4,:])
print(y_predict)
