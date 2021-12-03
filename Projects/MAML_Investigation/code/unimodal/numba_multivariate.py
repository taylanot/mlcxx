import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
from numba.experimental import jitclass
from tqdm import tqdm
from numba import float32, int32, float64

specs_multivariate = [('count',int32),('limit',int32),('dim',int32),\
            ('mean',float64[:,:]),('cov',float64[:,:]),('data',float64[:,:])]

@jitclass(specs_multivariate)
class Multivariate_Normal():
    def __init__(self, mean, cov, N=50):

        self.mean = mean.reshape(-1,1)
        self.cov = cov
        self.dim = self.mean.shape[0]
        self.count = -1
        self.limit = N
        self.data = np.empty((0,self.dim))
        self.main()

    def get_gaussian_random(self):
        m = 0
        while m == 0:
            m = round(np.random.random() * 100)
        numbers = np.random.random(int(m))
        summation = float(np.sum(numbers))
        gaussian = (summation - m/2) / math.sqrt(m/12.0)
        return gaussian

    def generate_known_gaussian(self):
        count = self.limit
        ret = []
        for i in range(count):
            current_vector = []
            for j in range(self.dim):
                g = self.get_gaussian_random()
                current_vector.append(g)
            ret.append( (current_vector) )
        return ret

    def main(self):
        known = self.generate_known_gaussian()
        [eigenvalues, eigenvectors] = np.linalg.eig(self.cov)
        l = np.diag(np.sqrt(eigenvalues))
        Q = eigenvectors @ l
        for i, data in enumerate(known):
            original = np.array(data).reshape(-1,1)
            tweaked = (Q @ original) + self.mean 
            self.data = np.append(self.data,tweaked.reshape(1,self.dim),axis=0)

    def sample(self, size=None):
        self.count += 1
        if self.count <= (self.limit-1):
            return self.data[self.count,:]
        else:
            raise Exception('Exceeded pre-generated points') 
#@nb.jit
#def sample():
#    for i in range(100000):
#        dist_a = Multivariate_Normal(np.ones(dim)*m,np.eye(dim)*c, Na)
#        a = dist_a.data
# 
#import time
#start = time.time()
#dim =1 
#m = 1
#c = 1
#Na = 50
#
#sample()
#    
#end = time.time()
#print(end-start)

