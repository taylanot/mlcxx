#External Stuff
import torch 
import matplotlib.pyplot as plt 
import os 
from argparse import Namespace
# Experiment Logger
from sacred import Experiment
# Internal Stuff
from task_generator import * 
from networks import LinearNetwork, NonlinearNetwork
from MAML import MAML, multitask

ex =  Experiment('MAML_Training')

@ex.config
def my_config():

    config = dict()
    seed = 24 
    torch.manual_seed(seed)
    problem = 'nonlinear'
    config['dim'] = 1
    config['std_y'] = 0.
    config['x_info'] = [0,2]
    if problem == 'linear':
        config['bias'] = True
        config['network'] = LinearNetwork#(config['dim'], 1, config['bias'])
        config['a_info'] = [0,1]
        config['pT'] = LinearTasks#(config['dim'], config['std_y'],\
                #config['a_info'])
    elif problem == 'nonlinear':
        config['n_neuron'] = 40
        config['activation_tag'] = 'relu'
        config['n_hidden'] = 2
        config['network'] = NonlinearNetwork#(
                #config['dim'], 1, config['n_hidden'],  config['n_neuron'],\
                #config['activation_tag'])
        config['a_info'] = [1,2]
        config['p_info'] = [0,2]
        config['pT'] = SineTasks#(config['dim'], config['std_y'],\
                #config['x_info'], config['a_info'], config['p_info'])
    config['alpha'] = 0.01
    config['beta'] = 0.001
    config['k'] = 5 
    config['task_batch_size'] = 10
    config['fo'] = False
    config['show_weights'] = False
    config['epochs'] = 70000

@ex.capture
def get_info(_run):
    return _run._id, _run.experiment_info["name"]

@ex.main
def main(config, seed, problem):
    conf = Namespace(**config)
    _id, _name = get_info()

    artifacts = os.path.join(_name, 'artifacts', _id)
    os.makedirs(artifacts)

    plot = False
    if conf.dim == 1:
        plot = True

    if problem == 'nonlinear':
        net = conf.network(conf.dim, 1, conf.n_hidden,  conf.n_neuron,\
                        conf.activation_tag)
        pT = conf.pT(conf.dim, conf.std_y,\
                conf.x_info, conf.a_info, conf.p_info)
    elif problem == 'linear':
        net = conf.network(conf.dim, 1, conf.bias)
        pT = conf.pT(conf.dim, conf.std_y, conf.x_info, conf.a_info)

    maml= multitask(network=net,  beta=conf.beta,\
            task_dist=pT, num_points_task=conf.k,\
            num_task_sample=conf.task_batch_size,\
            show_weights=conf.show_weights)

    maml.train(epochs=conf.epochs, path=artifacts)
    torch.save(net.state_dict(),artifacts+"/multitask.pt")

