import time
from numba.experimental import jitclass
from numba import float32, int32, float64
import numba as nb
import numpy as np

from sacred import Experiment
ex = Experiment('my_experiment')

@nb.jit()
def features(X):
    add = np.ones(X.shape[0]).reshape(-1,1)
    return np.ascontiguousarray(np.hstack((X,add)))

@ex.config
def my_config():
    alpha = 0.1

@ex.capture
def train(alpha):
    X = np.random.normal(0,1,(5,2))
    y = X
    np.ascontiguousarray(y, dtype=np.float64)
    model = Linear(X.shape[1])
    for i in range(1000000):
        model.fit(X,y)
        model.predict(X)

spec = [('dim', int32),('w', float64[:,:])]


@jitclass(spec)
class Linear():
    def __init__(self,dim):
        self.dim = dim

    def fit(self, X, Y):
        X = features(X)
        self.w = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(Y))
    
    def predict(self, X):
        X = features(X)
        return X.dot(self.w)

@ex.automain
def my_main():
   train() 
#train(0.1)

#model = Linear(3)

