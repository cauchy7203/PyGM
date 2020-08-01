from FGreyModels import FGM
import numpy as np
x = [15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5]
t = np.arange(len(x))+1
fgm = FGM(1)
fgm.fit(t, x)
y_predict = fgm.predict(t)
print(x)
print(y_predict)