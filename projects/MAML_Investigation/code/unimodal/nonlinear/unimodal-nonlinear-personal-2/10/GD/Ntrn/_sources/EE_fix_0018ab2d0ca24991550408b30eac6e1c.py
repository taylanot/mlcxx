# Science Stuff
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import numpy as np
# Models
from models import * 
from networks import NonlinearNetwork
# Experiment Logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
# Basic Python Stuff 
from argparse import Namespace
import copy
import os 
import warnings


ex =  Experiment('unimodal-nonlinear')

@ex.config
def my_config():
    NAME = "unimodal-nonlinear-personal-2"

    SUPPORTED_MODELS = ['Bayes', 'KernelRidge', 'MAML', 'GD']

    run_tag = 'Ntrn' 

    model_tag = 'GD'

    res = 20               # Resolution of the experiments! make 0 for just one 

    res_hyper = 10         # Resolution of the hypers!

    seed = 24               # KOBEEEEEE!

    config = {}
    
    config['dim'] = 10
    if run_tag == 'dim':
        config['dim'] = torch.arange(1, res).tolist()

    config['m_amplitude'] = 1.

    config['c_amplitude'] = 2.
    if run_tag == 'c_amplitude':
        config['c_amplitude'] = torch.linspace(0.001,10, res).tolist()

    config['m_phase'] = 0.

    config['c_phase'] = 2.
    if run_tag == 'c_phase':
        config['c_phase'] = torch.linspace(0.001,10, res).tolist()

    config['b'] = 2.
    if run_tag == 'b':
        config['b'] = torch.linspace(1,5, res).tolist()

    config['Na'] = 50
    config['Nz'] = 50

    config['std_y'] = 1
    if run_tag == 'std_y':
        config['std_y'] = torch.linspace(0,5, res).tolist()

    config['lr'] = torch.linspace(1e-4, 1, res_hyper).tolist()
    config['l'] = torch.linspace(1e-4, 5, res_hyper).tolist()
    
    config['alpha'] = torch.linspace(1e-4, 20, res_hyper).tolist()
    config['empty'] = []

    config['Ntrn'] = 5
    if run_tag == 'Ntrn':
        config['Ntrn'] = torch.arange(1, res).tolist()

    config['Ntst'] = 1000

    config['n_iter'] = 10
    if run_tag == 'n_iter':
        config['n_iter'] = torch.arange(0,100, res_hyper).tolist()

    if model_tag == 'MAML':
        config['model_ids'] = {1:1, 2:2, 10:3, 50:4}
        config['model_path'] = os.path.join('MAML_Training_noiseless', 'artifacts',\
                str(config['model_ids'][config['dim']]), 'model.pt')

    if run_tag == 'dim': 
        ex.observers.append(FileStorageObserver.create(NAME+'/'+str(model_tag)+'/'+str(run_tag)+'/'))
    else:
        ex.observers.append(FileStorageObserver.create(NAME+'/'+str(config['dim'])+'/'+str(model_tag)+'/'+str(run_tag)+'/'))


def set_seed(value):
    torch.manual_seed(value)

def sample_a(config, seed):
    set_seed(seed)
    var = Namespace(**config)
    tasks_amplitude = dist.MultivariateNormal(\
            torch.ones(var.dim)*var.m_amplitude, \
            torch.eye(var.dim)*var.c_amplitude)
    tasks_phase = dist.MultivariateNormal(torch.ones(var.dim)*var.m_phase, \
            torch.eye(var.dim)*var.c_phase)
    tasks = (tasks_amplitude.sample(torch.Size([var.Na])), \
            tasks_phase.sample(torch.Size([var.Na])))
    return tasks

def sample_x(config, seed):
    set_seed(seed)
    var = Namespace(**config)
    inputs = dist.MultivariateNormal(torch.zeros(var.dim), \
            torch.eye(var.dim)*var.b)
    train = inputs.sample(torch.Size([var.Na,var.Nz,var.Ntrn])) 
    test = inputs.sample(torch.Size([var.Na,var.Nz,var.Ntst])) 
    return train,test

def create_Z(config, seed):
    def f(a1,a2,x):
        return (torch.sin(x+a2) @ a1.reshape(-1,1))
    tasks = sample_a(config, seed)
    a1 ,a2 = tasks
    xs_train, xs_test = sample_x(config, seed)
    var = Namespace(**config)
    ys_train = []
    ys_test = []
    for i in range(var.Na):
        y_train = f(a1[i],a2[i],xs_train[i]).reshape(var.Nz, var.Ntrn, 1)
        y_test = f(a1[i],a2[i],xs_test[i]).reshape(var.Nz, var.Ntst, 1)
        y_train += torch.FloatTensor(y_train.size()).normal_(0, var.std_y)
        y_test += torch.FloatTensor(y_test.size()).normal_(0, var.std_y)
        ys_train.append(y_train)
        ys_test.append(y_test)
    return (tasks, (xs_train, torch.cat(ys_train).reshape(var.Na, var.Nz, var.Ntrn, 1)), \
            (xs_test, torch.cat(ys_test).reshape(var.Na, var.Nz, var.Ntst, 1)))

def EE_Bayes(config, seed, **kwargs):
    model = Bayes()
    tasks, trains, tests = create_Z(config, seed)
    xtests, ytests = tests
    xtrains, ytrains= trains
    var = Namespace(**config)
    a1, a2 = tasks
    loss_fn = torch.nn.MSELoss()
    ez = []
    ea = []
    for i in range(var.Na):
        model.fit((a1[i], a2[i]))
        for j in range(var.Nz):
            ez.append(loss_fn(model.predict(xtests[i][j]), ytests[i][j]))
        ea.append(torch.mean(torch.Tensor(ez)))
    return torch.mean(torch.Tensor(ea)).item()

def EE_KernelRidge(config, seed, hyper):
    tasks, trains, tests = create_Z(config, seed)
    xtests, ytests = tests
    xtrains, ytrains= trains
    var = Namespace(**config)
    a1, a2 = tasks
    loss_fn = torch.nn.MSELoss()
    for l in var.l:
        ee = []
        ez = []
        ea = []
        model = KernelRidge(lmbda=hyper, l=l)
        for i in range(var.Na):
            for j in range(var.Nz):
                model.fit((xtrains[i][j],ytrains[i][j]))
                ez.append(loss_fn(model.predict(xtests[i][j]), ytests[i][j]))
            ea.append(torch.mean(torch.Tensor(ez)))
        ee.append(torch.mean(torch.Tensor(ea)))
    ee = torch.min(torch.Tensor(ee))
    return ee.item()

def EE_MAML(config, seed, hyper):
    tasks, trains, tests = create_Z(config, seed)
    xtests, ytests = tests
    xtrains, ytrains= trains
    var = Namespace(**config)
    a1, a2 = tasks
    loss_fn = torch.nn.MSELoss()
    ez = []
    ea = []
    for i in range(var.Na):
        for j in range(var.Nz):
            model = NonlinearNetwork(in_feature=var.dim, n_neuron=40, out_feature=1, n_hidden=2, activation_tag='relu')
            model.load(var.model_path)
            #for name, param in model.state_dict().items():
            #    if name=='layers.output.bias':
            #        print(name, param)
            model.fit((xtrains[i][j],ytrains[i][j]), lr=hyper, load=False, n_iter=var.n_iter)
            ez.append(loss_fn(model.predict(xtests[i][j]), ytests[i][j]))
        ea.append(torch.mean(torch.Tensor(ez)))
    return torch.mean(torch.Tensor(ea)).item()

def EE_GD(config, seed, hyper):
    tasks, trains, tests = create_Z(config, seed)
    xtests, ytests = tests
    xtrains, ytrains= trains
    var = Namespace(**config)
    a1, a2 = tasks
    loss_fn = torch.nn.MSELoss()
    ez = []
    ea = []
    for i in range(var.Na):
        for j in range(var.Nz):
            model = NonlinearNetwork(in_feature=var.dim, n_neuron=40, out_feature=1, n_hidden=2, activation_tag='relu')
            #for name, param in model.state_dict().items():
            #    if name=='layers.output.bias':
            #        print(name, param)
            model.fit((xtrains[i][j],ytrains[i][j]), lr=hyper, load=False, n_iter=var.n_iter)
            ez.append(loss_fn(model.predict(xtests[i][j]), ytests[i][j]))
        ea.append(torch.mean(torch.Tensor(ez)))
    return torch.mean(torch.Tensor(ea)).item()


def EE(model_tag, run_tag, config, seed):
    run_config = config.copy()
    overall_ee = []
    err = []
    if model_tag == 'KernelRidge' or model_tag == 'MAML' or model_tag == 'GD':
        if model_tag == 'KernelRidge':
            name = 'alpha'
            Error = EE_KernelRidge 
        elif model_tag == 'MAML':
            name = 'lr'
            Error = EE_MAML
        elif model_tag == 'GD':
            name = 'lr'
            Error = EE_GD
        for hyper in config[name]:
            for value in config[run_tag]:
                print(value)
                run_config[run_tag] = value
                err.append(Error(run_config, seed, hyper))
            overall_ee.append(err)
        return np.array(overall_ee)
    else:
        for value in config[run_tag]:
            run_config[run_tag] = value
            err.append(EE_Bayes(run_config, seed))
        overall_ee.append(err)
        return np.array(overall_ee)
@ex.capture
def get_info(_run):
    return _run._id, _run.experiment_info["name"]

@ex.automain
def main(seed, config, model_tag, run_tag, SUPPORTED_MODELS):
    assert model_tag in SUPPORTED_MODELS
    result = EE(model_tag, run_tag, config, seed)
    _id, _name = get_info()
    filename = 'result-'+str(_id)+'-'+model_tag+run_tag+'.npy'
    np.save(filename, result)
    ex.add_artifact(filename)

