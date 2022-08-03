import numpy as np
import matplotlib.pyplot as plt 

from skfda.ml.regression import LinearRegression
from skfda.representation.basis import (FDataBasis, Monomial,Constant)

x_basis = Monomial(n_basis=3)
data = [[0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 2]]
        #[1, 0, 0],
        #[0, 0, 1],
        #[1, 1, 0],
        #[1, 1, 0]]
data = np.array(data)
#data = np.random.normal(0,1,(6,3))
#data = np.linspace(0,1)
x_fd = FDataBasis(x_basis, data)
x_fd.plot()
def f(x):
    return x**3 + np.random.normal(0,1,len(x))

y = f(np.arange(0,data.shape[0]))#, 32, 64, 128, 256]

model = LinearRegression()
model.fit(x_fd, y)
print(model.coef_[0])
print(model.intercept_)

y_plot = model.predict(x_fd)
print(model.score(x_fd, y))

#plt.plot(y_plot)
plt.show()
