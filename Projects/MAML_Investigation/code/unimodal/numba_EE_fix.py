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
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

ex =  Experiment('unimodal')

@ex.config
def my_config():
    NAME = "Unimodal_Experiments"

    SUPPORTED_MODELS = ['Bayes', 'SGD', 'Linear', 'Ridge']

    run_tag = 'std_y' 

    model_tag = 'SGD'
    
    res = 100               # Resolution of the experiments!

    res_hyper = 10          # Resolution of the hypers!


    config = {}
    config['seed'] = 24 # KOBEEEEEE!
    
    config['dim'] = 1.
    if run_tag == 'dim':
        config['dim'] = np.arange(1, 20, res).tolist()

    config['m'] = 0.
    if run_tag == 'm':
        config['m'] = np.linspace(0,10, res).tolist()

    config['c'] = 1.
    if run_tag == 'c':
        config['c'] = np.linspace(0,10, res).tolist()

    config['b'] = 1.
    if run_tag == 'b':
        config['b'] = np.linspace(1,5, res).tolist()

    config['Na'] = 1000.
    config['Nz'] = 1000.

    config['std_y'] = 1
    if run_tag == 'std_y':
        config['std_y'] = np.linspace(0,5, res).tolist()

    config['lr'] = np.linspace(1e-4, 1, res_hyper).tolist()
    
    config['alpha'] = np.linspace(1e-4, 5, res_hyper).tolist()

    config['Ntrn'] = 1
    if run_tag == 'Ntrn':
        config['Ntrn'] = np.arange(1, 20, res).tolist()

    config['Ntst'] = 5000.

    config['n_iter'] = 1
    if run_tag == 'n_iter':
        config['n_iter'] = np.arange(0,100, res_hyper).tolist()



    numba_config = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )
    
    # The typed-dict can be used from the interpreter.
    for k, v in config.items():
        if type(v) is list:
            numba_config[k] = np.array(v, dtype='f8')
        else:
            numba_config[k] = np.array([v], dtype='f8')
 
@nb.njit
def set_seed(value):
    np.random.seed(value)

@nb.njit
def error(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

@nb.njit
def EE_Bayes(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na):
    model = Bayes(c,std_y)
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
def EE_Ridge(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, alpha):
    model = Ridge(alpha)
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
def EE_SGD(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, lr, n_iter):
    model = SGD(lr,n_iter)
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

@ex.automain
def main(run_tag, numba_config):

    #var = Namespace(**config)
    #assert var.model in var.SUPPORTED_MODELS, "Model is not supported!"
    #EE = []
    #if var.model == "Bayes":
    #    err = []
    #    set_seed(var.seed)
    #    for var.__dict__[run_tag] in config[run_tag]:
    #        err.append(EE_Bayes(var.dim, var.m, var.c, var.std_y, var.b, \
    #                var.Ntrn, var.Ntst, var.Nz, var.Na))
    #    EE.append({run_tag:err})

    #elif var.model == "Linear":
    #    err = []
    #    set_seed(var.seed)
    #    for var.__dict__[run_tag] in config[run_tag]:
    #        err.append(EE_Linear(var.dim, var.m, var.c, var.std_y, var.b, var.Ntrn,\
    #                var.Ntst, var.Nz, var.Na))
    #    EE.append({run_tag:err})

    #elif var.model == "Ridge":
    #    for var.__dict__['alpha'] in config['alpha']:
    #        err = []
    #        set_seed(var.seed)
    #        for var.__dict__[run_tag] in config[run_tag]:
    #            err.append(EE_Ridge(var.dim, var.m, var.c, var.std_y, var.b, \
    #                    var.Ntrn, var.Ntst, var.Nz, var.Na, var.alpha))
    #        EE.append({'alpha':var.alpha, run_tag:err})

    #elif var.model == "SGD":
    #    for var.__dict__['lr'] in config['lr']:
    #        err = []
    #        set_seed(var.seed)
    #        for var.__dict__[run_tag] in config[run_tag]:
    #            err.append(EE_SGD(var.dim, var.m, var.c, var.std_y, var.b, \
    #                    var.Ntrn, var.Ntst, var.Nz, var.Na, var.lr, var.n_iter))
    #        EE.append({'lr':var.lr, run_tag:err})
    
    return numba_config


