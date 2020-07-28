from CommonOperation.Accumulation import Accumulation
from CommonOperation.GreyModels import BernoulliGM
import numpy as np
x = [15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5]
t = np.arange(len(x))+1
N = 2
Bgm = BernoulliGM()
Bgm.fit(N, x)
y_predict = Bgm.predict(t)
print(x)
print(y_predict)