import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy 
from sacred import Experiment
from models import Linear, Bayes, SGD, Ridge
from tqdm import tqdm
import torch
from torch.distributions import multivariate_normal, normal, uniform
import time

start  = time.time()
seed = 24

torch.manual_seed(seed)
np.random.seed(seed)


def sample_a(dim,m,c):
        dist = multivariate_normal.MultivariateNormal(torch.ones(dim)*m, torch.eye(dim)*c)
        return dist.sample()

def sample_x(dim, b, N):
    dist = uniform.Uniform(0, b)
    return dist.sample(torch.Size([N,dim]))

def sample_y(a,x,std_y):
    dist = normal.Normal(0, std_y) 
    dist.sample(torch.Size([x.shape[0],1]))

    return torch.matmul(x,a.T)

def sample_Z(dim, a, b, std_y, N):
    x = sample_x(dim, b, N)
    y = sample_y(a, x, std_y)
    return x, y

def error(y,yp):
    return torch.matmul((yp-y).T,(yp-y)).mean()

params = dict()
params['dim'] = 1
params['m'] = 0
params['c'] = 1
params['std_y'] = 1
params['b'] = 1
params['Ntrn'] =5 
params['Ntst'] = 1000
params['Nz'] = 1
params['Na'] = 1 
params['lr'] = 1 
params['alpha'] = 1



def exp_err(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, model):
    eea = 0
    for j in range(Na):
        eez = 0
        a = sample_a(dim, m, c)
        print(a)
        for i in range(Nz):
            xtrn, ytrn = sample_Z(dim, a, b, std_y, Ntrn)
            xtst, ytst = sample_Z(dim, a, b, std_y, Ntst)

            model.fit(xtrn,ytrn)
            eez += (error(ytst, torch.Tensor(model.predict(xtst).reshape(-1,1))))/Nz
        eea += (eez)/Na
    return eea

exp_err(params['dim'], params['m'], params['c'], params['std_y'], params['b'],\
            params['Ntrn'], params['Ntst'], params['Nz'], params['Na'], Bayes())
end = time.time()
print(end-start)

#
#
#ex = Experiment('Experiment')
#
#@ex.config
#def my_config():
#    params = dict()
#    params['dim'] = 1
#    params['m'] = 0
#    params['c'] = np.array([0.,0.5,1.,5.])
#    params['std_y'] = np.linspace(0,3,10)
#    params['b'] = np.linspace(1,5,5)
#    params['Ntrn'] = [1,2,5,10]
#    params['Ntst'] = 5000
#    params['Nz'] = 1
#    params['Na'] = 1 
#    params['lr'] = np.linspace(1e-3,1,5)
#    params['alpha'] = np.linspace(0,5,10)
#
#@ex.named_config
#def config_Bayes():
#    model_base = Bayes
#
#@ex.named_config
#def config_Linear():
#    model_base = Linear
#
#@ex.named_config
#def config_Ridge():
#    model_base = Ridge
#
#@ex.named_config
#def config_MAML():
#    model_base = SGD
#
#@ex.capture
#def get_error_single(params, model_base):
#    err = np.zeros((len(params['c']),len(params['std_y']),len(params['b']),len(params['Ntrn'])))
#    for i,c in (enumerate(tqdm(params['c']))):
#        for j,std_y in enumerate(params['std_y']):
#            for k,b in enumerate(params['b']):
#                for l,Ntrn in enumerate(params['Ntrn']):
#                    if model_base is Bayes:
#                        model = model_base(c, std_y)
#                    else:
#                        model = model_base()
#                    err[i,j,k,l] = exp_err(params['dim'], params['m'], c, std_y, b, Ntrn, params['Ntst'], params['Nz'], params['Na'], model)
#    return err.tolist()
#
#@ex.capture
#def get_error_multiple(params, model_base):
#    if model_base is SGD:
#        err = np.zeros((len(params['c']),len(params['std_y']),len(params['b']),len(params['Ntrn']), len(params['lr'])))
#    elif model_base is Ridge:
#        err = np.zeros((len(params['c']),len(params['std_y']),len(params['b']),len(params['Ntrn']), len(params['alpha'])))
#
#    for i,c in (enumerate(tqdm(params['c']))):
#        for j,std_y in enumerate(params['std_y']):
#            for k,b in enumerate(params['b']):
#                for l,Ntrn in enumerate(params['Ntrn']):
#                    if model_base is Ridge:
#                        for m,alpha in enumerate(params['alpha']):
#                            model = model_base(alpha=alpha)
#                            err[i,j,k,l,m] = exp_err(params['dim'], params['m'], c, std_y, b, Ntrn, params['Ntst'], params['Nz'], params['Na'], model)
#                    if model_base is SGD:
#                        for m,lr in enumerate(params['lr']):
#                            model = model_base(lr=lr)
#                            err[i,j,k,l,m] = exp_err(params['dim'], params['m'], c, std_y, b, Ntrn, params['Ntst'], params['Nz'], params['Na'], model)
#    return err.tolist()
#
#@ex.automain
#def main(_config,model_base):
#    if model_base == Linear or model_base == Bayes:
#        return get_error_single()
#    else:
#        return get_error_multiple()
#
