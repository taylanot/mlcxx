import numpy as np
import matplotlib.pyplot as plt
def pairwise_l2_distance(x,y):
    D = -2 * x @  y.T + np.sum(y**2, axis=1) + np.sum(x**2, axis=1)[:,np.newaxis]
    D[D<0] = 0.
    return D

class sine():
    def __init__(self,l=1):
        self.l = l
    def __call__(self, x, xp=None):
        if np.any(xp == None):
            xp = x
        alpha = pairwise_l2_distance(x,xp)
        return np.sin(alpha)/np.pi*(alpha)

class KernelRidge():
    def __init__(self, lmbda, kernel):
        self.lmbda= lmbda
        self.kernel = kernel

    def fit(self, X, Y):
        self.alpha = np.linalg.inv(self.kernel(X,X) + self.lmbda*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def predict(self, X):
        return (self.kernel(X, self.X)).dot(self.alpha)

X = np.linspace(-5,5,100).reshape(-1,1)
Y = np.sin(X) + np.random.normal(0,1,X.shape)

model = KernelRidge(0, sine())
model.fit(X,Y)
xplt = np.linspace(-5, 5, 100).reshape(-1,1)
plt.plot(xplt, model.predict(xplt))
plt.show()

