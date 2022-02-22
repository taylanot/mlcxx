import numpy as np
import matplotlib.pyplot as plt
np.random.seed(24)
class sample_a():
    def __init__(self, dim, mean, cov):
        self.dim = dim
        self.dist = np.random.multivariate_normal
    def sample(self, N):
        return self.dist(np.ones(dim)*mean, np.eye(dim)*cov, N)

class sample_data():
    def __init__(self, dim, cov):
        self.dist = np.random.multivariate_normal
        self.dim = dim
        self.cov = cov
    def sample(self,a, N):
        x = self.dist(np.ones(dim)*0, np.eye(dim)*self.cov, N)
        #y = x @ a.T + np.random.normal(0,1,(N,1))
        y = a*np.sin(x)  + np.random.normal(0,1,(N,1))
        return x, y

dim = 1
mean = 0
cov = 1
C = []
N = 10
lmbda = 0.5
xpred = np.linspace(-4,4,200).reshape(-1,1)
a_dist = sample_a(dim, mean, cov)
data_dist = sample_data(dim, cov=1)
for i in range(100):
    x,y = data_dist.sample(a_dist.sample(1),N)
    X = np.hstack((x, np.ones((N,1))))
    x_bars.append(xq
    C.append(X.T.dot(X)+lmbda*np.eye(dim+1))
print(sum(C)/len(C))



