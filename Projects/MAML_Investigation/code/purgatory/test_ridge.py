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
alpha = 0.
Ntrn = 5
Ntst = 1000
std_y = 1.
b = 1.
model = Ridge(alpha)
#model = Linear()
a = np.zeros(dim)
xtrn = np.random.uniform(0, b, (Ntrn,dim))
ytrn = (np.random.normal(0, std_y) + xtrn.dot(a)).reshape(-1,1)
xtst = np.random.uniform(0, b, (Ntst,dim))
ytst = (np.random.normal(0, std_y) + xtst.dot(a)).reshape(-1,1)
model.fit(xtrn,ytrn)
print(model.w)
