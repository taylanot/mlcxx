# Science Stuff
import numpy as np
import matplotlib.pyplot as plt
# Hardcore Numba
import numba as nb
from numba_models import Bayes, Linear, Ridge, SGD
from numba.core.errors import NumbaPerformanceWarning
from numba_multivariate import *
# Basic Python Stuff 
from argparse import Namespace
import copy
import os 
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@nb.njit
def set_seed(value):
    np.random.seed(value)

@nb.njit
def error(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

@nb.njit
def EE_Linear(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na):
    model = Linear()
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

@nb.njit
def get_error(std_ys = np.linspace(0,10,100)):
    err = []
    set_seed(24)
    for std_y in std_ys:
        err.append(EE_Linear(dim=1, m=0, c=1, std_y=std_y, b=1, Ntrn=1,\
                Ntst=1000, Nz=10, Na=10))

    return err

plt.plot(np.linspace(0,10, 100), get_error())
plt.savefig('pinv.pdf')

