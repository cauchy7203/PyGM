from Modules.NIPGreyModels import NIPNGM
import numpy as np
x = np.array([8.6, 9.32, 10.184, 11.221, 12.465, 13.958])

t = np.arange(len(x))+1
nipngm = NIPNGM()
nipngm.fit(t[:4], x[:4])
y_predict = nipngm.predict(t)
print(x)
print(y_predict)
