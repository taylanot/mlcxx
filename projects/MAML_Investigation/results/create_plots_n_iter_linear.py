# File Handling Stuff
import json
import os
from glob import glob
# Stats from statistics import mean, pstdev
import itertools as it
# Plotting Stuff
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cycler
import numpy as np
import tikzplotlib
import os
# Define your plot cycle color
n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

Experiment_Directory = 'unimodal-linear-niter'
run_file = 'run.json'
config_file = 'config.json'
Ntrns = np.array([1,2,10])
models = ['MAML', 'SGD']
dims = np.array([1,2,10])
n_iters = np.arange(1,11)
combs = it.product(Ntrns,n_iters)
ids = dict()
for i, comb in enumerate(combs):
    ids[str(i+1)] = comb

def load(run, config):
    with open(run) as json_file:
        run_info = json.load(json_file)
    with open(config) as json_file:
        config_info = json.load(json_file)
    return config_info, run_info

def find_best(x, ys, model, config_info, n_iter):
    ymeans = [np.mean(res) for res in ys]
    #ymeans = [res[3] for res in ys]
    select = 3#ymeans.index(min(ymeans))
    y = ys[select]
    label_ext = '-$\eta$:'+str(round(config_info['config']['lr'][select],4))+'-n_iter:'+str(n_iter)
    if model == 'SGD':
        model_tag = 'GD'
    else:
        model_tag = model
    tag = model_tag+label_ext
    return x, y, tag, select

   
def get_results(dims, models, ids,exp_dir=Experiment_Directory, run_tag='c'):
    for model in models:
        for dim in dims:
            for ID, vals in ids.items():
                n_iter = vals[1]
                Ntrn = vals[0]
                run = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), run_file)
                config = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), config_file)
                config_info, run_info = load(run, config)
                x = config_info['config'][run_tag]
                ys = run_info['result']
                x,y,tag,select = find_best(x,ys,model, config_info,n_iter)
                #print(np.mean(y), tag)
                if dim == 1 and Ntrn== 10:
                    print(np.mean(y), tag,select)

get_results(dims, models, ids)
