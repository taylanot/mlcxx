import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
from numba.experimental import jitclass
from numba import float32, int32, float64
from tqdm import tqdm
#@nb.jit()
#def get_gaussian_random():
#    m = 0
#    while m == 0:
#        m = round(np.random.random() * 100)
#    numbers = np.random.random(int(m))
#    summation = float(np.sum(numbers))
#    gaussian = (summation - m/2) / math.sqrt(m/12.0)
#
#    return gaussian
#
#@nb.njit
#def generate_known_gaussian(dimensions):
#    count = 1000
#
#    ret = []
#    for i in range(count):
#        current_vector = []
#        for j in range(dimensions):
#            g = get_gaussian_random()
#            current_vector.append(g)
#
#        ret.append( (current_vector) )
#
#    return ret
#
#@nb.jit
#def main():
#    known = generate_known_gaussian(2)
#    target_mean = np.array([ [1.0], [5.0]])
#    target_cov  = np.array([[  1.0, 0.7], 
#                             [  0.7, 0.6]])
#
#    [eigenvalues, eigenvectors] = np.linalg.eig(target_cov)
#    l = np.diag(np.sqrt(eigenvalues))
#    Q = eigenvectors @ l
#    x1_tweaked = []
#    x2_tweaked = []
#    tweaked_all = []
#    for data in known:
#        original = np.array(data).reshape(-1,1)
#        tweaked = (Q @ original) + target_mean
#        x1_tweaked.append(tweaked[0])
#        x2_tweaked.append(tweaked[1])
#        tweaked_all.append( tweaked )
#    return x1_tweaked, x2_tweaked
#
#x1, x2 = main()

#generate_known_gaussian(2)
#plt.scatter(x1, x2)
#plt.axis([-6, 10, -6, 10])
#plt.hlines(0, -6, 10)
#plt.vlines(0, -6, 10)
#plt.savefig('plot.pdf')

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
            self.data = np.append(self.data,tweaked.reshape(1,2),axis=0)

    def sample(self, size=None):
        self.count += 1
        if self.count <= (self.limit-1):
            return self.data[self.count,:]
        else:
            raise Exception('Exceeded pre-generated points') 

import time
start = time.time()
dim = 2
for i in tqdm(range(100000)):
    dist = Multivariate_Normal(np.ones(dim),np.eye(dim))
end = time.time()
print(end-start)

start = time.time()
dim = 2
for i in tqdm(range(100000)):
    dist = np.random.multivariate_normal(np.ones(dim),np.eye(dim),(50,1))
end = time.time()
print(end-start)


#plt.scatter(dist.data[:,0], dist.data[:,1])
#plt.savefig('plot.pdf')

