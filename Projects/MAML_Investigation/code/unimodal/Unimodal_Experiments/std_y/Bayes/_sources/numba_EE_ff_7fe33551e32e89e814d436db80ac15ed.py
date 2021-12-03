# Science Stuff
import numpy as np
import matplotlib.pyplot as plt
# Hardcore Numba
import numba as nb
from numba_models import Bayes, Linear, Ridge, SGD
from numba.core.errors import NumbaPerformanceWarning
from numba_multivariate import *
# Experiment Logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
# Basic Python Stuff 
from argparse import Namespace
import copy
import os 
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

ex =  Experiment('unimodal')

@ex.config
def configuration():
    
    NAME = "Unimodal_Experiments"

    run_tag = 'std_y' 
    
    res = 100               # Resolution of the experiments!

    res_hyper = 10          # Resolution of the hypers!

    config = {}
    config['seed'] = 24 # KOBEEEEEE!
    
    config['dim'] = 1
    if run_tag == 'dim':
        config['dim'] = np.arange(1, 20, res).tolist()

    config['m'] = 0 
    if run_tag == 'm':
        config['m'] = np.linspace(0,10, res).tolist()

    config['c'] = 1
    if run_tag == 'c':
        config['c'] = np.linspace(0,10, res).tolist()

    config['b'] = 1
    if run_tag == 'b':
        config['b'] = np.linspace(1,5, res).tolist()

    config['Na'] = 1000
    config['Nz'] = 1000

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

    config['model'] = 'SGD'
    config['SUPPORTED_MODELS'] = ['Bayes', 'SGD', 'Linear', 'Ridge']
    ex.observers.append(FileStorageObserver.create(NAME+'/'+str(run_tag)+'/'+str(config['model'])+'/'))

@nb.njit
def set_seed(value):
    np.random.seed(value)

@nb.njit
def error(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

@nb.njit
def EE_Bayes(run_tag, dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, seed):
    EE = []
    var = dict()
    var[str(dim)] = dim
    var[str(m)] = m  
    var[str(c)] = c
    var[str(std_y)] = std_y
    var[str(b)] = b  
    var[str(Ntrn)] = Ntrn
    var[str(Ntst)] = Ntst
    var[str(Nz)] = Nz
    var[str(Na)] = Na
    var_init = var

    for var[run_tag] in var_init:
        ee = []
        set_seed(seed)
        model = Bayes(var['c'],var['std_y'])
        dist = Multivariate_Normal(np.ones(var['dim'])*m,np.eye(var['dim'])*var['c'],var['Na']*10)
        eea = 0
        for j in range(var['Na']):
            eez = 0
            a = dist.sample()
            for i in range(var['Nz']):
                xtrn = np.random.uniform(0, var['b'], (var['Ntrn'],var['dim']))
                ytrn = (np.random.normal(0, var['std_y']) + xtrn.dot(a)).reshape(-1,1)
                xtst = np.random.uniform(0, var['b'], (var['Ntrn'],var['dim']))
                ytst = (np.random.normal(0, var['std_y']) + xtst.dot(a)).reshape(-1,1)
                model.fit(xtrn,ytrn)
                eez += np.mean(error(ytst, model.predict(xtst).reshape(-1,1)))
            eea += eez / Nz
    ee.append(eea / Na)
    EE.append({run_tag:err})
    return EE

@nb.njit
def EE_Linear(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, seed):
    set_seed(seed)
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
def EE_Ridge(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, alpha, seed):
    set_seed(seed)
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
def EE_SGD(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, lr, n_iter, seed):
    set_seed(seed)
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
def main(run_tag, config):
    var = Namespace(**config)
    if var.model == "Bayes":
        res = EE_Bayes(run_tag, var.dim, var.m, var.c, var.std_y, var.b, \
                    var.Ntrn, var.Ntst, var.Nz, var.Na, var.seed)

    elif var.model == "Linear":
        err = []
        set_seed(var.seed)
        for var.__dict__[run_tag] in config[run_tag]:
            err.append(EE_Linear(var.dim, var.m, var.c, var.std_y, var.b, var.Ntrn,\
                    var.Ntst, var.Nz, var.Na))
        EE.append({run_tag:err})

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
    
    return EE


