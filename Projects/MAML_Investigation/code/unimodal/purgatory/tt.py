import numpy as np
import numba as nb
from numba import njit, prange
import time 

#def sample(dim, m, c, b, Na, Nz, std_y, Ntrn, Ntst):
#    a = np.random.multivariate_normal(np.ones(dim)*m,np.eye(dim)*c, Na)
#    xtst =  np.random.uniform(0,b,(Nz,Ntst,dim))
#    xtrn =  np.random.uniform(0,b,(Nz,Ntrn,dim))
#    test_set = []
#    train_set = []
#    for i in range(Na):
#        ytst = (xtst @ a[i].reshape(-1,1)) 
#        ytst += np.random.normal(0,std_y,(ytst.shape))
#        ytrn = (xtrn @ a[i].reshape(-1,1)) 
#        ytrn += np.random.normal(0,std_y,(ytrn.shape))
#        test_set.append((xtst, ytst))
#        train_set.append((xtrn, ytrn))
#    return train_set, test_set

#@nb.jit()
#def sample_a(dim,m,c):
#    return np.random.normal(0,1)
#
#@nb.jit()
#def sample_x(dim, b, N):
#    return np.random.uniform(0,b,(N,dim))
#
#@nb.jit()
#def sample_y(a,x,std_y):
#    return x @ a.reshape(-1,1)
#
#@nb.jit()
#def sample_Z(dim, a, b, std_y, N):
#    x = sample_x(dim, b, N)
#    y = sample_y(a, x, std_y)
#    return x, y
@nb.jit()
def get_gaussian_random():
    m = 0
    while m == 0:
        m = round(np.random.random() * 100)
    numbers = np.random.random(int(m))
    summation = float(np.sum(numbers))
    gaussian = (summation - m/2) / math.sqrt(m/12.0)

    return gaussian
@nb.jit()
def generate_known_gaussian(dimensions):
    count = 1000

    ret = []
    for i in xrange(count):
        current_vector = []
        for j in xrange(dimensions):
            g = get_gaussian_random()
            current_vector.append(g)

        ret.append( tuple(current_vector) )

    return ret

#@nb.jit()
def sample(dim, m, c, b, Na, Nz, std_y, Ntrn, Ntst):
    for i in range(Na):
        a = 
        for j in range(Nz):
            xtst = np.random.normal(0,1)

start = time.time()
sample(1,1,1,1,10000,10000,1,10,1000)
end= time.time()
print(end-start)
