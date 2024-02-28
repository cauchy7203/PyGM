from Modules.NIPGreyModels import NIPNDGM
import numpy as np
x = np.array([8.6, 9.32, 10.184, 11.221, 12.465, 13.958])
t = np.arange(len(x))+1
nipndgm = NIPNDGM()
nipndgm.fit(t[:4], x[:4])
y_predict = nipndgm.predict(t[:4])
print(x)
print(y_predict)
