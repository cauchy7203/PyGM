from Modules.NIPGreyModels import NIPNDGM
import numpy as np
x = [15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5]
t = np.arange(len(x))+1
nipndgm = NIPNDGM(1)
nipndgm.fit(t, x)
y_predict = nipndgm.predict(t)
print(x)
print(y_predict)