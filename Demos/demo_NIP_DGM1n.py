from CommonOperation.NIPGreyModels import NIPDGMN
import numpy as np

x = [[560823, 542386, 604834, 591248, 583031, 640636, 575688, 689637, 570790, 519574, 614677],
     [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3],
     [135.6, 140.2, 140.1, 146.9, 144, 143, 133.3, 135.7, 125.8, 98.5, 99.8],
     [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5],
     [54.2, 54.9, 54.8, 56.3, 54.5, 54.6, 54.9, 54.8, 49.3, 41.5, 48.9]]
x = np.array(x).T
t = np.arange(len(np.array(x)[0:, 0])) + 1

nipdgmm = NIPDGMN(1)
x1 = nipdgmm.fit(t, x)
t = t.reshape([-1, 1])
t_x = np.concatenate((t, x1), axis=1)

y_predict = nipdgmm.predict(t_x)
print(x)
print(y_predict)
