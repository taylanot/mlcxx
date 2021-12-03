import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict
from sacred import Experiment
from argparse import Namespace
# The Dict.empty() constructs a typed dictionary.
# The key and value typed must be explicitly declared.
ex = Experiment('run')
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
            


# Here's a function that expects a typed-dict as the argument
@njit
def move(numba_config):
    var = Namespace(**numba_config)
    # inplace operations on the arrays
    for k,v in numba_config.items():
        v += 1
    return numba_config
# Call move(d) to inplace update the arrays in the typed-dict.

@ex.automain
def main(numba_config):
    print(move(numba_config))

