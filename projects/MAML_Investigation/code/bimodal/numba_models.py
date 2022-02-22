import time
from numba.experimental import jitclass
from numba import float32, int32, float64
import numba as nb
import numpy as np

spec_linear = [('w', float64[:,:])]
spec_ridge = [('alpha',float64),('w', float64[:,:])]
spec_sgd = [('lr',float64),('n_iter',int32),('w', float64[:,:])]
spec_bayes_textbook = [('m',float64),('beta',float64),('alpha',float64),('std_y',float64),('SN', float64[:,:]),('mN', float64[:,:])]
spec_bayes = [('mN', float64[:,:])]


@nb.njit(cache=True)
def features(X):
    add = np.ones(X.shape[0]).reshape(-1,1)
    return np.ascontiguousarray(np.hstack((X,add)))

@nb.njit(cache=True)
def set_bias(X, noise=False):
    if noise:
        add = np.random.normal(0,0.1,(1,1))
    else:
        add = np.zeros(1).reshape(-1,1)
    return np.ascontiguousarray(np.vstack((X,add)))



@jitclass(spec_linear)
class Linear():
    def __init__(self):
        pass

    def fit(self, X, Y):
        X = features(X)
        self. w = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(Y))
    
    def predict(self, X):
        X = features(X)
        return X.dot(self.w)

@jitclass(spec_ridge)
class Ridge():
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, Y):
        X = features(X)
        self. w = np.linalg.pinv(X.T.dot(X) + np.eye(X.shape[1])*self.alpha).dot(X.T.dot(Y))
    
    def predict(self, X):
        X = features(X)
        return X.dot(self.w)

@jitclass(spec_ridge)
class GeneralRidge():
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, Y, set):
        X = features(X)
        set = set_bias(set) 
        self. w = np.linalg.pinv(X.T.dot(X) + np.eye(X.shape[1])*self.alpha).dot((X.T.dot(Y))+set*self.alpha)
    
    def predict(self, X):
        X = features(X)
        return X.dot(self.w)



@jitclass(spec_sgd)
class SGD():
    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter

    def init_weights(self, d):
        self.w = np.random.normal(0,1,(d+1,1))

    def set_weights(self,w):
        self.w = set_bias(w)

    def fit(self, X, Y, set):
        X = features(X)
        N = X.shape[0]
        self.set_weights(set)
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = 2 * X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def fit_random(self, X, Y):
        X = features(X)
        N = X.shape[0]
        self.init_weights(X.shape[1])
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def predict(self, X):
        X = features(X)
        return X.dot(self.w)

@jitclass(spec_sgd)
class MAML():
    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter

    def init_weights(self, d):
        self.w = np.random.normal(0,1,(d+1,1))

    def set_weights(self,w):
        self.w = set_bias(w, True)

    def fit(self, X, Y, set):
        X = features(X)
        N = X.shape[0]
        self.set_weights(set)
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = 2 * X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def fit_random(self, X, Y):
        X = features(X)
        N = X.shape[0]
        self.init_weights(X.shape[1])
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def predict(self, X):
        X = features(X)
        return X.dot(self.w)

@jitclass(spec_bayes_textbook)
class Bayes_textbook():
    def __init__(self, m, alpha, std_y):
        jitter = 1.e-12
        self.alpha = alpha 
        self.std_y= std_y
        if self.alpha == 0.:
            self.alpha += jitter
        if self.std_y == 0.:
            self.beta = 1. / (self.std_y**2+jitter)
        else:
            self.beta = 1. / (self.std_y**2+jitter)
        self.m = m

    def fit(self, X,y):
        S0_inv = (1./self.alpha) * np.eye(X.shape[1])
        SN_inv =  S0_inv + self.beta * np.dot(X.T, X)
        self.SN = np.linalg.inv(SN_inv)
        self.mN = (np.dot(self.SN,(S0_inv.dot(self.m*np.ones(X.shape[1]).reshape(-1,1))+self.beta*(X.T).dot(y))))

    def predict(self,X):
        return X.dot(self.mN)

@jitclass(spec_bayes)
class Bayes():
    def __init__(self):
        pass

    def fit(self, a):
        self.mN = a

    def predict(self,X):
        return X.dot(self.mN)


