import numpy as np
import matplotlib.pyplot as plt
np.random.seed(24)

def sample_test(a=100, phi=np.pi, N=50, noise=True, sort=True, width=5):
    #x = np.random.uniform(-width, width, (N,1))
    x = np.random.normal(0, 2, (N,1))
    if sort:
        x = np.sort(x,axis=0)
        if noise:
            y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,0.1,(N,1))
        else:
            y = np.multiply(a, np.sin(x+phi)) 
    return x, y 

def pairwise_l2_distance(x,y):
    D = -2 * x @  y.T + np.sum(y**2, axis=1) + \
            np.sum(x**2, axis=1)[:,np.newaxis]
    D[D<0] = 0.
    return D

class rbf():
    def __init__(self,l=1):
        self.l = l

    def __call__(self, x, xp=None):
        if np.any(xp == None):
            xp = x
        alpha = pairwise_l2_distance(x,xp)
        return np.exp(-alpha/self.l**2)

def hat(X,Y,x,l=1):
    kernel = rbf(l)
    sim = kernel(X, x)
    #return (sim * Y).sum(axis=0) / sim.sum(axis=0)
    print((sim / sim.sum(axis=0)).shape)
    return (sim / sim.sum(axis=0)).T.dot(Y)
    #return np.dot(kernel(x,X)/kernel(x,X).sum(axis=0),Y)
X, Y = sample_test(1,0,20, noise=True, sort=True)

plt.plot(X,Y)
x = np.linspace(-3,3,1000).reshape(-1,1)
ls = np.linspace(0.1,50,50)
for l in ls:
    plt.plot(X,hat(X,Y,X,l))
plt.plot(x, np.sin(x),'r',':')
    #print(hat(X,Y,x,l))
plt.show()




