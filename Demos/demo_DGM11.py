from Modules.GreyModels import DGM
import numpy as np
x = [8.6, 9.32, 10.184, 11.221, 12.465, 13.958]
t = np.arange(len(x))+1
dgm = DGM()
dgm.fit(t[:4], x[:4])
y_predict = dgm.predict(t)
print(x)
print(y_predict)
