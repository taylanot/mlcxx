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

ex =  Experiment('unimodal-tests')

@ex.config
def my_config():
    NAME = "Unimodal_Experiments_dim"

    SUPPORTED_MODELS = ['Bayes', 'SGD', 'Linear', 'Ridge']

    run_tag = 'std_y' 

    model_tag = 'Bayes'
    
    res = 20 # Resolution of the experiments!

    res_hyper = 20         # Resolution of the hypers!

    seed = 24               # KOBEEEEEE!

    config = {}
    
    config['dim'] = 1 
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

    config['Na'] = 10
    config['Nz'] = 10

    config['std_y'] = 1
    if run_tag == 'std_y':
        config['std_y'] = np.linspace(0,5, res).tolist()

    config['lr'] = np.linspace(1e-4, 1, res_hyper).tolist()
    
    config['alpha'] = np.linspace(1e-4, 20, res_hyper).tolist()

    config['Ntrn'] = 1 
    if run_tag == 'Ntrn':
        config['Ntrn'] = np.arange(1, 50).tolist()

    config['Ntst'] = 1000

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
 
#    ex.observers.append(FileStorageObserver.create(NAME+'/'+str(config['dim'])+'/'+str(model_tag)+'/'+str(run_tag)+'/'))


@nb.njit(cache=True)
def set_seed(value):
    np.random.seed(value)

@nb.njit(cache=True)
def error(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

@nb.njit(cache=True)
def batch(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]

#@nb.njit
#def EE_Bayes(a_dist, xtrn_dist, xtst_dist, model_tag, run_tag, numba_config, numba_config_runner):
#    var = numba_config 
#    run = numba_config_runner
#    overall_ee = [ ] 
#    err = [ ]
#    for value in var[run_tag]:
#        run[run_tag] = np.array([value], dtype='f8')
#        model = Bayes(float(run['m'][0]),float(run['c'][0]), float(run['std_y'][0]))
#        a_gen = batch(a_dist, 1)
#        eea = 0
#        for a in a_gen:
#            eez = 0
#            trn_gen = batch(xtrn_dist, int(run['Ntrn'][0]))
#            tst_gen = batch(xtst_dist, int(run['Ntst'][0]))
#            for xtrn in trn_gen:
#                ytrn = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntrn'][0]),1)) + xtrn.dot(a.T)).reshape(-1,1)
#                xtst = next(tst_gen)
#                ytst = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntst'][0]),1)) + xtst.dot(a.T)).reshape(-1,1)
#                model.fit(xtrn,ytrn)
#                eez += np.mean((error(ytst, model.predict(xtst).reshape(-1,1))))
#            eea += eez / float(run['Nz'][0])
#        err.append(eea / float(run['Na'][0]))
#    overall_ee.append(err)
#    return overall_ee

#@nb.njit
#def EE_Bayes(a_dist, xtrn_dist, xtst_dist, model_tag, run_tag, numba_config, numba_config_runner):
#    var = numba_config 
#    run = numba_config_runner
#    overall_ee = [ ] 
#    for alpha in var['alpha']:
#        err = [ ]
#        for value in var[run_tag]:
#            run[run_tag] = np.array([value], dtype='f8')
#            model = Ridge(alpha)
#            a_gen = batch(a_dist, 1)
#            eea = 0
#            for a in a_gen:
#                eez = 0
#                trn_gen = batch(xtrn_dist, int(run['Ntrn'][0]))
#                tst_gen = batch(xtst_dist, int(run['Ntst'][0]))
#                for xtrn in trn_gen:
#                    ytrn = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntrn'][0]),1)) + xtrn.dot(a.T)).reshape(-1,1)
#                    xtst = next(tst_gen)
#                    ytst = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntst'][0]),1)) + xtst.dot(a.T)).reshape(-1,1)
#                    model.fit(xtrn,ytrn)
#                    eez += np.mean((error(ytst, model.predict(xtst).reshape(-1,1))))
#                eea += eez / float(run['Nz'][0])
#            err.append(eea / float(run['Na'][0]))
#        overall_ee.append(err)
#    return overall_ee


#@nb.njit(cache=True)
#def EE_Bayes(a_dist, xtrn_dist, xtst_dist, model_tag, run_tag, numba_config, numba_config_runner):
#    var = numba_config 
#    run = numba_config_runner
#    overall_ee = [ ] 
#    for lr in var['lr']:
#        err = [ ]
#        for value in var[run_tag]:
#            run[run_tag] = np.array([value], dtype='f8')
#            model = SGD(lr, int(run['n_iter'][0]))
#            a_gen = batch(a_dist, 1)
#            eea = 0
#            for a in a_gen:
#                eez = 0
#                trn_gen = batch(xtrn_dist, int(run['Ntrn'][0]))
#                tst_gen = batch(xtst_dist, int(run['Ntst'][0]))
#                for xtrn in trn_gen:
#                    ytrn = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntrn'][0]),1)) + xtrn.dot(a.T)).reshape(-1,1)
#                    xtst = next(tst_gen)
#                    ytst = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntst'][0]),1)) + xtst.dot(a.T)).reshape(-1,1)
#                    model.fit(xtrn,ytrn,np.ascontiguousarray(a).reshape(-1,1))
#                    eez += np.mean((error(ytst, model.predict(xtst).reshape(-1,1))))
#                eea += eez / float(run['Nz'][0])
#            err.append(eea / float(run['Na'][0]))
#        overall_ee.append(err)
#    return overall_ee

@nb.njit(cache=True)
def EE_Bayes(seed, model_tag, run_tag, numba_config, numba_config_runner):
    var = numba_config 
    run = numba_config_runner
    overall_ee = [ ] 
    xtrn_dist = Multivariate_Normal(np.zeros(int(run['dim'][0])),np.eye(int(run['dim'][0]))*run['b'][0], int(run['Ntrn'][0]*run['Nz'][0])).data
    xtst_dist = Multivariate_Normal(np.zeros(int(run['dim'][0])),np.eye(int(run['dim'][0]))*run['b'][0], int(run['Ntst'][0]*run['Nz'][0])).data
    a_dist = Multivariate_Normal(np.zeros(int(run['dim'][0]))*run['m'][0],np.eye(int(run['dim'][0]))*run['c'][0], int(run['Na'][0])).data
    for lr in var['lr']:
        err = [ ]
        for value in var[run_tag]:
            set_seed(seed)
            run[run_tag] = np.array([value], dtype='f8')
            model = SGD(lr, int(run['n_iter'][0]))
            a_gen = batch(a_dist, 1)
            eea = 0
            for a in a_gen:
                eez = 0
                trn_gen = batch(xtrn_dist, int(run['Ntrn'][0]))
                tst_gen = batch(xtst_dist, int(run['Ntst'][0]))
                for xtrn in trn_gen:
                    ytrn = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntrn'][0]),1)) + xtrn.dot(a.T)).reshape(-1,1)
                    xtst = next(tst_gen)
                    ytst = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntst'][0]),1)) + xtst.dot(a.T)).reshape(-1,1)
                    model.fit(xtrn,ytrn,np.ascontiguousarray(a).reshape(-1,1))
                    eez += np.mean((error(ytst, model.predict(xtst).reshape(-1,1))))
                eea += eez / float(run['Nz'][0])
            err.append(eea / float(run['Na'][0]))
        overall_ee.append(err)
    return overall_ee

#@nb.njit(cache=True)
#def EE_Bayes(seed, model_tag, run_tag, numba_config, numba_config_runner):
#    var = numba_config 
#    run = numba_config_runner
#    overall_ee = [ ] 
#    xtrn_dist = Multivariate_Normal(np.zeros(int(run['dim'][0])),np.eye(int(run['dim'][0]))*run['b'][0], int(run['Ntrn'][0]*run['Nz'][0]*run['Na'][0]*len(var['lr'])))
#    xtst_dist = Multivariate_Normal(np.zeros(int(run['dim'][0])),np.eye(int(run['dim'][0]))*run['b'][0], int(run['Ntst'][0]*run['Nz'][0]*run['Na'][0]*len(var['lr'])))
#    a_dist = Multivariate_Normal(np.zeros(int(run['dim'][0]))*run['m'][0],np.eye(int(run['dim'][0]))*run['c'][0], int(run['Na'][0]*len(var['lr'])))
#    for lr in var['lr']:
#        err = [ ]
#        for value in var[run_tag]:
#            set_seed(seed)
#            run[run_tag] = np.array([value], dtype='f8')
#            model = SGD(lr, int(run['n_iter'][0]))
#            eea = 0
#            for _ in range(int(run['Na'][0])):
#                eez = 0
#                a = a_dist.sample(1)
#                for _ in range(int(run['Nz'][0])):
#                    xtrn = xtrn_dist.sample(int(run['Ntrn'][0]))
#                    ytrn = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntrn'][0]),1)) + xtrn.dot(a.T)).reshape(-1,1)
#                    xtst = xtst_dist.sample(int(run['Ntst'][0]))
#                    ytst = (np.random.normal(0, float(run['std_y'][0]), (int(run['Ntst'][0]),1)) + xtst.dot(a.T)).reshape(-1,1)
#                    model.fit(xtrn,ytrn,np.ascontiguousarray(a).reshape(-1,1))
#                    eez += np.mean((error(ytst, model.predict(xtst).reshape(-1,1))))
#                eea += eez / float(run['Nz'][0])
#            err.append(eea / float(run['Na'][0]))
#        overall_ee.append(err)
#    return overall_ee


@ex.automain
def main(seed, config, model_tag, run_tag, numba_config, numba_config_runner, SUPPORTED_MODELS):
    #assert model_tag in SUPPORTED_MODELS, "Model is not supported!"
    #set_seed(seed)
    #xtrn_dist = np.random.multivariate_normal(np.zeros(config['dim']),np.eye(config['dim'])*config['b'], (config['Ntrn']*config['Nz']))
    #xtst_dist = np.random.multivariate_normal(np.zeros(config['dim']),np.eye(config['dim'])*config['b'], (config['Ntst']*config['Nz']))
    #a_dist = np.random.multivariate_normal(np.zeros(config['dim'])*config['m'],np.eye(config['dim'])*config['c'], (config['Na']))
    #if model_tag == "Bayes":
    #    res = EE_Bayes(a_dist, xtrn_dist, xtst_dist, model_tag, run_tag, numba_config, numba_config_runner)
    #    plt.plot(config[run_tag],res[0])
    #    plt.savefig('test_bayes.pdf')
    #return res

    assert model_tag in SUPPORTED_MODELS, "Model is not supported!"
    if model_tag == "Bayes":
        res = EE_Bayes(seed, model_tag, run_tag, numba_config, numba_config_runner)
    return res


