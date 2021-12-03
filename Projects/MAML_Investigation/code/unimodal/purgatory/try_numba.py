import time
from numba.experimental import jitclass
from numba import float32, int32, float64
import numba as nb
import numpy as np

from sacred import Experiment
ex = Experiment('my_experiment')

@ex.config
def my_config():
    alpha = 0.1
    lr = 0.01
    

@ex.capture
@nb.jit()
def train(alpha,lr):
    X = np.random.normal(0,1,(5,2))
    X = features(X)
    y = np.random.normal(0,1,(5,1))
    np.ascontiguousarray(y)
    model = Bayes(0.1,5.0)
    for i in range(1000):
        for j in range(1000):
            model.fit(X,y)
            model.predict(X)

spec = [('w', float64[:,:])]
spec_ridge = [('alpha',float64),('w', float64[:,:])]
spec_sgd = [('lr',float64),('n_iter',int32),('w', float64[:,:])]
spec_bayes = [('beta',float64),('alpha',float64),('std_y',float64),('SN', float64[:,:]),('mN', float64[:,:])]


@nb.jit()
def features(X):
    add = np.ones(X.shape[0]).reshape(-1,1)
    return np.ascontiguousarray(np.hstack((X,add)))

@jitclass(spec)
class Linear():
    def __init__(self):
        pass

    def fit(self, X, Y):
        self. w = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(Y))
    
    def predict(self, X):
        return X.dot(self.w)

@jitclass(spec_ridge)
class Ridge():
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, Y):
        self. w = np.linalg.pinv(X.T.dot(X) + np.eye(X.shape[1])*self.alpha).dot(X.T.dot(Y))
    
    def predict(self, X):
        return X.dot(self.w)

@jitclass(spec_sgd)
class SGD():
    def __init__(self, lr=0.001, n_iter=1):
        self.lr = lr
        self.n_iter = n_iter

    def init_weights(self, d):
        self.w = np.random.normal(0,1,(d,1))

    def fit(self, X, Y):
        N = X.shape[0]
        self.init_weights(X.shape[1])
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def predict(self, X):
        return X.dot(self.w)

@jitclass(spec_bayes)
class Bayes():
    def __init__(self, alpha=0., std_y= 5):
        jitter = 1.e-12
        self.alpha = alpha + jitter
        self.beta = 1. / (std_y**2+jitter)

    def fit(self, X,y):
        S0_inv = (1./self.alpha) * np.eye(X.shape[1])
        SN_inv =  S0_inv + self.beta * np.dot(X.T, X)
        self.SN = np.linalg.inv(SN_inv)
        self.mN  = self.beta * (np.dot(self.SN, (X.T).dot(y)))

    def predict(self,X):
        return X.dot(self.mN)

@ex.automain
def my_main():
   train() 
#train(0.1)

#model = Linear(3)

