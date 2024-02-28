from Modules.GreyModels import GM
from CommonOperations.Generate_report import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = [15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5]
t = np.arange(len(x)) + 1
x0_train = x[:4]
x0_test = x[4:]
t_train = t[:4]
t_test = t[4:]
gm = GM()
gm.fit(t, x)
report = generate_report()
res = report.generate_report_gm(gm, t_train, x0_train, t_test, x0_test)
plt.show()