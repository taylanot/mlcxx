#External Stuff
import torch 
import matplotlib.pyplot as plt 
import os 
from argparse import Namespace
# Experiment Logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
# Internal Stuff
from task_generator import * 
from networks import LinearNetwork, NonlinearNetwork
from MAML import MAML

ex =  Experiment('MAML_Training')

@ex.config
def my_config():

    config = dict()
    seed = 24 
    problem = 'nonlinear'

    config['dim'] = 1
    config['x_info'] = [0,2]
    if problem == 'linear':
        config['bias'] = True
        config['network'] = LinearNetwork(config['dim'], 1, config['bias'])
        config['a_info'] = [4,1]
        config['pT'] = LinearTasks(config['dim'], config['a_info'])
    elif problem == 'nonlinear':
        config['n_neuron'] = 40
        config['activation_tag'] = 'tanh'
        config['n_hidden'] = 1
        config['network'] = NonlinearNetwork(
                config['dim'], 1, config['n_hidden'],  config['n_neuron'],\
                config['activation_tag'])
        config['a_info'] = [1,2]
        config['p_info'] = [0,1]
        config['pT'] = SineTasks(config['dim'], config['x_info'], \
                config['a_info'], config['p_info'])
    config['alpha'] = 0.1
    config['beta'] = 0.01
    config['k'] = 5 
    config['task_batch_size'] = 50
    config['fo'] = False
    config['show_weights'] = False
    config['epochs'] = 3000

    NAME = os.path.join("MAML_Training",problem,str(config['dim']))
    ex.observers.append(FileStorageObserver(NAME))

@ex.capture
def get_info(_run):
    return _run._id, _run.experiment_info["name"]

@ex.automain
def main(config, seed, NAME):
    conf = Namespace(**config)
    _id, _ = get_info()
    artifacts = os.path.join(NAME,'artifacts',_id)
    os.makedirs(artifacts)
    torch.manual_seed(seed)

    
    maml= MAML(network=conf.network, alpha=conf.alpha, beta=conf.beta,\
            task_dist=conf.pT, num_points_task=conf.k,\
            num_task_sample=conf.task_batch_size, first_order=conf.fo,\
            show_weights=conf.show_weights)

    maml.train(epochs=conf.epochs, path=artifacts)
    torch.save(conf.network.state_dict(),artifacts+"/model.pt")


