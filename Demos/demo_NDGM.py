from Modules.GreyModels import NDGM
import numpy as np
x = np.array([8.6, 9.32, 10.184, 11.221, 12.465, 13.958])
t = np.arange(len(x))+1
ndgm = NDGM()
t_train = t[:4]
x_train = x[:4]
ndgm.fit(t_train, x_train)
y_predict = ndgm.predict(t)
print(x)
print(y_predict)
