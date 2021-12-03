import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from numba_models import Bayes, Linear, Ridge, SGD
import numba as nb
from argparse import Namespace
from sacred import Experiment
from sacred.observers import FileStorageObserver
from numba_multivariate import *
import copy
import os 

@nb.njit
def set_seed(value):
    np.random.seed(value)

@nb.jit
def error(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

@nb.njit
def exp_err(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na):
    model=Linear()
    dist = Multivariate_Normal(np.ones(dim)*m,np.eye(dim)*c,Na*10)
    eea = 0
    for j in range(Na):
        eez = 0
        a = dist.sample()
        for i in range(Nz):
            xtrn = np.random.uniform(0, b, (Ntrn,dim))
            ytrn = (np.random.normal(0, std_y) + xtrn.dot(a)).reshape(-1,1)
            xtst = np.random.uniform(0, b, (Ntrn,dim))
            ytst = (np.random.normal(0, std_y) + xtst.dot(a)).reshape(-1,1)
            model.fit(xtrn,ytrn)
            eez += np.mean(error(ytst, model.predict(xtst).reshape(-1,1)))
        eea += eez / Nz
    return eea / Na

import time 
set_seed(24)
start = time.time()
std_ys = np.linspace(0,10)
err = []
for std_y in std_ys:
    err.append(exp_err(3,1,1,std_y,1,10,10,1000,10))
end = time.time()
print(end-start)
import matplotlib.pyplot as plt
plt.plot(std_ys, err)
plt.savefig('test.pdf')
