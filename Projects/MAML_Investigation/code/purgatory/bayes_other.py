# Science Stuff
import numpy as np
import matplotlib.pyplot as plt
# Hardcore Numba
import numba as nb
from numba_models import Bayes, Linear, Ridge, SGD
from numba.core.errors import NumbaPerformanceWarning
from numba_multivariate import *
from numba.core import types
from numba.typed import Dict
# Experiment Logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
# Basic Python Stuff 
from argparse import Namespace
import copy
import os 
import warnings
# Filter some annoying warnings

@nb.njit
def error(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

np.random.seed(24)
dim = 1 
Ntrn = 1
Ntst = 1000
std_y = 1.
b = 1.
m = 1.
c = 1.
model = Bayes(m, c, std_y)
a = np.random.multivariate_normal(np.ones(dim)*m,np.eye(dim)*c,1).reshape(-1,1)
print(a)
xtrn = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*b, Ntrn)
ytrn = (np.random.normal(0, std_y,(Ntrn,1)) + xtrn.dot(a)).reshape(-1,1)
ytrn = (np.random.normal(0, std_y,(Ntrn,1)) + xtrn.dot(a)).reshape(-1,1)
xtst = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*b, Ntst)
ytst = (np.random.normal(0, std_y, (Ntst, 1)) + xtst.dot(a)).reshape(-1,1)
ytrn = (np.random.normal(0, std_y,(Ntrn,1)) + xtrn.dot(a)).reshape(-1,1)
model.fit(xtrn,ytrn)
print(model.mN)
print(error(ytst, model.predict(xtst)))
import matplotlib.pyplot as plt
plt.scatter(xtrn, ytrn)
x = np.linspace(-1,1).reshape(-1,dim)
plt.plot(x, model.predict(x),color='k')
plt.savefig('look_bayes.pdf')
