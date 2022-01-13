# File Handling Stuff
import json
import os
from glob import glob
# Stats
from statistics import mean, pstdev
# Plotting Stuff
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cycler
import numpy as np

# Define your plot cycle color
n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

Experiment_Directory = 'Unimodal_Experiments'
run_file = 'run.json'
config_file = 'config.json'
Ntrns = np.array([1,2,10,50])
models = ['Bayes', 'SGD', 'Linear', 'Ridge']
np.where(Ntrns==1)

def get_results(exp_dir=Experiment_Directory, model='Linear', run_tag='std_y', dim=1, Ntrn=1):
    
    ID = np.where(Ntrns==Ntrn)[0][0]+1
    run = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), run_file)
    config = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), config_file)

    with open(run) as json_file:
        run_info = json.load(json_file)
    with open(config) as json_file:
        config_info = json.load(json_file)

    x = config_info['config'][run_tag]
    ys = run_info['result']

    if len(ys) == 1:
        y = ys[0]
        label_ext = ''
        print(y)
    else:
        ymeans = [mean(res) for res in run_info['result']]
        select = ymeans.index(min(ymeans))
        y = ys[select]
        print(y)
        if model == 'Ridge':
            label_ext = '-alpha:'+str((config_info['config']['alpha'][select]))
        elif model == 'SGD':
            label_ext = '-lr:'+str((config_info['config']['lr'][select]))
    tag = model+label_ext
    return x, y, tag

def logplot(ax, x, y, tag):
    ax.semilogy(x,y, label=tag)

def plot(ax, x, y, tag):
    ax.plot(x,y, label=tag)


def add_legend(ax,inside=True):
    if inside:
        ax.legend(frameon=False)
    else:
        ax.legend(frameon=False,bbox_to_anchor =(0.5,-0.3), loc='lower center',ncol=4)

def add_label(ax, run_tag):
    ax.set_xlabel(run_tag)
    ax.set_ylabel("EE")

fig, ax = plt.subplots(figsize=(6,6))
#x,y,tag = get_results(model='SGD')
#plot(ax, x, y, tag)

def plot_result(run_tag, dim, Ntrn, log=False, ex='Linear'):

    for model in models:
        if model != ex:
            x,y,tag = get_results(model=model, run_tag=run_tag,dim=dim, Ntrn=Ntrn)
            if log:
                logplot(ax, x, y, tag)
            else:
                plot(ax, x, y, tag)
            add_legend(ax)
            add_label(ax,run_tag)
    plt.show()

plot_result('std_y', dim=10, Ntrn=10, log=False)

### NEED to write some other function for n_iter






