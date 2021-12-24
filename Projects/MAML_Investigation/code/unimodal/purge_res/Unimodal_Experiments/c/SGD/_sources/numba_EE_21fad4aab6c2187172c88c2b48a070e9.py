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

    res_hyper = 20         # Resolution of the hypers!

    seed = 24               # KOBEEEEEE!

    config = {}
    
    config['dim'] = 2.
    if run_tag == 'dim':
        config['dim'] = np.arange(1, 50).tolist()

    config['m'] = 0.
    if run_tag == 'm':
        config['m'] = np.linspace(0,10, res).tolist()

    config['c'] = 1.
    if run_tag == 'c':
        config['c'] = np.linspace(0,10, res).tolist()

    config['b'] = 1.
    if run_tag == 'b':
        config['b'] = np.linspace(1,5, res).tolist()

    config['Na'] = 100.
    config['Nz'] = 100.

    config['std_y'] = 1
    if run_tag == 'std_y':
        config['std_y'] = np.linspace(0,5, res).tolist()

    config['lr'] = np.linspace(1e-4, 1, res_hyper).tolist()
    
    config['alpha'] = np.linspace(1e-4, 5, res_hyper).tolist()

    config['Ntrn'] = 1
    if run_tag == 'Ntrn':
        config['Ntrn'] = np.arange(1, 50).tolist()

    config['Ntst'] = 5000.

    config['n_iter'] = 1
    if run_tag == 'n_iter':
        config['n_iter'] = np.arange(0,100, res_hyper).tolist()

# Stupid Numba, but fasssssstttt ...

    numba_config = Dict.empty(key_type=types.unicode_type, \
            value_type=types.float64[:])
    numba_config_runner = Dict.empty(key_type=types.unicode_type, \
            value_type=types.float64[:])
    
    for k, v in config.items():
        if type(v) is list:
            numba_config[k] = np.array(v, dtype='f8')
            numba_config_runner[k] = np.array(v, dtype='f8')
        else:
            numba_config[k] = np.array([v], dtype='f8')
            numba_config_runner[k] = np.array([v], dtype='f8')
 
    ex.observers.append(FileStorageObserver.create(NAME+'/'+str(run_tag)+'/'+str(model_tag)+'/'))

@nb.njit
def set_seed(value):
    np.random.seed(value)

@nb.njit
def error(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

@nb.njit
def EE_Linear(seed, model_tag, run_tag, numba_config, numba_config_runner):
    var = numba_config 
    run = numba_config_runner
    err = [ ]
    overall_ee= [ ]
    for value in var[run_tag]:
        run[run_tag] = np.array([value], dtype='f8')
        set_seed(seed)
        model = Linear()
        dist = Multivariate_Normal(np.ones(int(run['dim'][0]))*run['m'][0], \
                np.eye(int(run['dim'][0]))*run['c'], run['Na'][0]*10)
        eea = 0
        for j in range(int(run['Na'][0])):
            eez = 0
            a = dist.sample()
            for i in range(int(run['Nz'][0])):
                xtrn = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                ytrn = (np.random.normal(0, float(run['std_y'][0])) + xtrn.dot(a)).reshape(-1,1)
                xtst = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                ytst = (np.random.normal(0, float(run['std_y'][0])) + xtst.dot(a)).reshape(-1,1)
                model.fit(xtrn,ytrn)
                eez += np.mean(error(ytst, model.predict(xtst).reshape(-1,1)))
            eea += eez / run['Nz'][0]
        err.append(eea / run['Na'][0])
    overall_ee.append(err)
    return overall_ee

@nb.njit
def EE_Bayes(seed, model_tag, run_tag, numba_config, numba_config_runner):
    var = numba_config 
    run = numba_config_runner
    overall_ee = [ ] 
    err = [ ]
    for value in var[run_tag]:
        run[run_tag] = np.array([value], dtype='f8')
        set_seed(seed)
        model = Bayes(float(run['c'][0]), float(run['std_y'][0]))
        dist = Multivariate_Normal(np.ones(int(run['dim'][0]))*run['m'][0], \
                np.eye(int(run['dim'][0]))*run['c'], run['Na'][0]*10)
        eea = 0
        for j in range(int(run['Na'][0])):
            eez = 0
            a = dist.sample()
            for i in range(int(run['Nz'][0])):
                xtrn = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                ytrn = (np.random.normal(0, float(run['std_y'][0])) + xtrn.dot(a)).reshape(-1,1)
                xtst = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                ytst = (np.random.normal(0, float(run['std_y'][0])) + xtst.dot(a)).reshape(-1,1)
                model.fit(xtrn,ytrn)
                eez += np.mean(error(ytst, model.predict(xtst).reshape(-1,1)))
            eea += eez / run['Nz'][0]
        err.append(eea / run['Na'][0])
    overall_ee.append(err)
    return overall_ee

def EE_Ridge(seed, model_tag, run_tag, numba_config, numba_config_runner):
    var = numba_config 
    run = numba_config_runner
    overall_ee = []
    for alpha in var['alpha']:
        err = [ ]
        for value in var[run_tag]:
            run[run_tag] = np.array([value], dtype='f8')
            set_seed(seed)
            model = Ridge(alpha)
            dist = Multivariate_Normal(np.ones(int(run['dim'][0]))*run['m'][0], \
                    np.eye(int(run['dim'][0]))*run['c'], run['Na'][0]*10)
            eea = 0
            for j in range(int(run['Na'][0])):
                eez = 0
                a = dist.sample()
                for i in range(int(run['Nz'][0])):
                    xtrn = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                    ytrn = (np.random.normal(0, float(run['std_y'][0])) + xtrn.dot(a)).reshape(-1,1)
                    xtst = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                    ytst = (np.random.normal(0, float(run['std_y'][0])) + xtst.dot(a)).reshape(-1,1)
                    model.fit(xtrn,ytrn)
                    eez += np.mean(error(ytst, model.predict(xtst).reshape(-1,1)))
                eea += eez / run['Nz'][0]
            err.append(eea / run['Na'][0])
        overall_ee.append(err)
    return overall_ee 

def EE_SGD(seed, model_tag, run_tag, numba_config, numba_config_runner):
    var = numba_config 
    run = numba_config_runner
    overall_ee = []
    for lr in var['lr']:
        err = [ ]
        for value in var[run_tag]:
            run[run_tag] = np.array([value], dtype='f8')
            set_seed(seed)
            model = SGD(lr, int(run['n_iter'][0]))
            dist = Multivariate_Normal(np.ones(int(run['dim'][0]))*run['m'][0], \
                    np.eye(int(run['dim'][0]))*run['c'], run['Na'][0]*10)
            eea = 0
            for j in range(int(run['Na'][0])):
                eez = 0
                a = dist.sample()
                for i in range(int(run['Nz'][0])):
                    xtrn = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                    ytrn = (np.random.normal(0, float(run['std_y'][0])) + xtrn.dot(a)).reshape(-1,1)
                    xtst = np.random.uniform(0, float(run['b'][0]), (int(run['Ntrn'][0]),int(run['dim'][0])))
                    ytst = (np.random.normal(0, float(run['std_y'][0])) + xtst.dot(a)).reshape(-1,1)
                    model.fit(xtrn,ytrn)
                    eez += np.mean(error(ytst, model.predict(xtst).reshape(-1,1)))
                eea += eez / run['Nz'][0]
            err.append(eea / run['Na'][0])
        overall_ee.append(err)
    return overall_ee




@ex.automain
def main(seed, config, model_tag, run_tag, numba_config, numba_config_runner, SUPPORTED_MODELS):

    assert model_tag in SUPPORTED_MODELS, "Model is not supported!"
    if model_tag == "Bayes":
        res = EE_Bayes(seed, model_tag, run_tag, numba_config, numba_config_runner)

    elif model_tag == "Linear":
        res = EE_Linear(seed, model_tag, run_tag, numba_config, numba_config_runner)

    elif model_tag == "Ridge":
        res = EE_Ridge(seed, model_tag, run_tag, numba_config, numba_config_runner)

    elif model_tag == "SGD":
        res = EE_SGD(seed, model_tag, run_tag, numba_config, numba_config_runner)
    
    return res


