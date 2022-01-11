import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
from numba.experimental import jitclass
from tqdm import tqdm
from numba import float32, int32, float64, prange

specs_multivariate = [('count',int32),('end',int32),('limit',int32),('dim',int32),\
            ('mean',float64[:,:]),('cov',float64[:,:]),('data',float64[:,:])]

#@jitclass(specs_multivariate)
class Multivariate_Normal():
    def __init__(self, mean, cov, N=50):

        self.mean = mean.reshape(-1,1)
        self.cov = cov
        self.dim = self.mean.shape[0]
        self.count = -1
        self.limit = N
        self.data = np.empty((0,self.dim))
        self.end = 0
        self.main()

    #def get_gaussian_random(self):
    #    m = 0
    #    while m == 0:
    #        m = round(np.random.random() * 100)
    #    numbers = np.random.random(int(m))
    #    summation = float(np.sum(numbers))
    #    gaussian = (summation - m/2) / math.sqrt(m/12.0)
    #    return gaussian

    #def generate_known_gaussian(self):
    #    count = self.limit
    #    ret = []
    #    for i in range(count):
    #        current_vector = []
    #        for j in range(self.dim):
    #            g = self.get_gaussian_random()
    #            current_vector.append(g)
    #        ret.append( (current_vector) )
    #    return ret

    def main(self):
        #known = self.generate_known_gaussian()
        known = np.random.normal(0,1,(self.limit,self.dim))
        [eigenvalues, eigenvectors] = np.linalg.eig(self.cov)
        l = np.diag(np.sqrt(eigenvalues))
        Q = np.ascontiguousarray(eigenvectors) @ np.ascontiguousarray(l)
        for i, original in enumerate(known):
            tweaked = (np.ascontiguousarray(Q) @ np.ascontiguousarray(original).reshape(-1,1)) + self.mean 
            self.data = np.append(self.data,tweaked.reshape(1,self.dim),axis=0)

    def sample(self, size=0):
        self.count = self.end
        if self.count <= (self.limit-1):
            self.end = self.count+size
            return self.data[self.count:self.count+size,:]
        else:
            raise Exception('Exceeded pre-generated points') 

#@nb.njit(cache=True)
#def get_gaussian_random():
#        m = 0
#        while m == 0:
#            m = round(np.random.random() * 100)
#        numbers = np.random.random(int(m))
#        summation = float(np.sum(numbers))
#        gaussian = (summation - m/2) / math.sqrt(m/12.0)
#        return gaussian
#
#@nb.njit(cache=True)
#def generate_known_gaussian(dim,N):
#    ret = []
#    for i in range(N):
#        current_vector = []
#        for j in range(dim):
#            g = get_gaussian_random()
#            current_vector.append(g)
#        ret.append( (current_vector) )
#    return ret
#
#
#@nb.njit(cache=True)
#def multivariate_normal(mean, cov, N):
#    mean = mean.reshape(-1,1)
#    dim = mean.shape[1]
#    data = np.empty((0,dim))
#    known = generate_known_gaussian(dim,N) 
#    [eigenvalues, eigenvectors] = np.linalg.eig(cov)
#    l = np.diag(np.sqrt(eigenvalues))
#    Q = eigenvectors @ l
#    for i, dat in enumerate(known):
#        original = np.array(dat).reshape(-1,1)
#        tweaked = (Q @ original) + mean 
#        data = np.append(data,tweaked.reshape(1,dim),axis=0)
#    return data
#

#@nb.njit(cache=True)
#def multivariate_normal2(mean, cov, N):
#    mean = mean.reshape(-1,1)
#    dim = mean.shape[1]
#    data = np.empty((0,dim))
#    known = np.random.normal(0,1,(N,dim))
#    [eigenvalues, eigenvectors] = np.linalg.eig(cov)
#    l = np.diag(np.sqrt(eigenvalues))
#    Q = np.ascontiguousarray(eigenvectors) @ np.ascontiguousarray(l)
#    for i, original in enumerate(known):
#        tweaked = (np.ascontiguousarray(Q) @ np.ascontiguousarray(original)) + mean 
#        data = np.append(data,tweaked.reshape(1,dim),axis=0)
#    return data
#
#@nb.njit(cache=True)
#def sample_class():
#    for i in range(NO):
#        for _ in range(NO):
#            dist_a = Multivariate_Normal(np.ones(dim)*m,np.eye(dim)*c, Na).data
##@nb.njit(cache=True)
#def sample_func():
#    for i in range(NO):
#        for _ in range(NO):
#            sample = np.random.multivariate_normal(np.ones(dim)*m,np.eye(dim)*c, Na)
#
#@nb.njit(cache=True)#,parallel=True)
#def sample_func2():
#    for i in prange(NO):
#        for _ in prange(NO):
#            sample = multivariate_normal2(np.ones(dim)*m,np.eye(dim)*c, Na)


#import time
#dim = 100 
#m = 1
#c = 1
#Na = 100
#NO = 100
#start = time.time()
#dist_a = Multivariate_Normal(np.ones(dim)*m,np.eye(dim)*c, Na).data
#end = time.time()
#print(end-start)
#start = time.time()
#dist_a = np.random.multivariate_normal(np.ones(dim)*m,np.eye(dim)*c, Na)
#end = time.time()
#print(end-start)

#start = time.time()
#sample_func()
#end = time.time()
#print(end-start)
#
#start = time.time()
#sample_class()
#end = time.time()
#print(end-start)
#
#start = time.time()
#sample_class()
#end = time.time()
#print(end-start)
#
#
###@nb.jit
###def sample():
###    for i in range(100000):
###        dist_a = Multivariate_Normal(np.ones(dim)*m,np.eye(dim)*c, Na)
###        a = dist_a.data
## 
###start = time.time()
###dim =1 
###m = 1
###c = 1
###Na = 50
###
###sample()
###    
###end = time.time()
###print(end-start)
###
###start = time.time()
###np.random.multivariate_normal(np.ones(dim)*m,np.eye(dim)*c,Na*100000)
###end = time.time()
###print(end-start)
##
##
