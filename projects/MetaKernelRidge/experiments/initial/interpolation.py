import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from csaps import csaps

np.random.seed(24)

N = 200
std = 0.1

x = np.linspace(-20, 20,N) 

y = np.sin(x) + np.random.normal(0,std,N)
xnew = np.linspace(-20, 20,100)


ynew, smooth = csaps(x, y, xnew)

plt.plot(x, y, 'o', xnew, ynew, '-')

plt.show()
