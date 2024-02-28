from Modules.FGreyModels import FBernoulliGM
import numpy as np
x = [5,6,4,7]
t = np.arange(len(x))+1
fBgm = FBernoulliGM()
fBgm.fit(x)
y_predict = fBgm.predict(t)
print(x)
print(y_predict)

