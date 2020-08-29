from CommonOperation.GreyModels import NDGM
import numpy as np
x = [15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5]
t = np.arange(len(x))+1
ndgm = NDGM()
ndgm.fit(t, x)
y_predict = ndgm.predict(t)
print(x)
print(y_predict)