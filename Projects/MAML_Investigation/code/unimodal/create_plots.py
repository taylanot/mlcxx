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

# Define your plot cycle color
n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

Experiment_Directory = 'Unimodal_Experiments'
run_file = 'run.json'
config_file = 'config.json'

def get_results(exp_dir=Experiment_Directory, model='Linear', run_tag='std_y', ID=1):

    run = os.path.join(exp_dir, run_tag, model, str(ID), run_file)
    config = os.path.join(exp_dir, run_tag, model, str(ID), config_file)

    with open(run) as json_file:
        run_info = json.load(json_file)
    with open(config) as json_file:
        config_info = json.load(json_file)

    x = config_info['config'][run_tag]
    ys = run_info['result']

    if len(ys) == 1:
        y = ys[0][run_tag]
        label_ext = ''
    else:
        ymeans = [mean(res[run_tag]) for res in run_info['result']]
        select = ymeans.index(min(ymeans))
        y = ys[select][run_tag]
        if model == 'Ridge':
            label_ext = '-alpha:'+str(round(ys[select]['alpha']))
        elif model == 'SGD':
            label_ext = '-lr:'+str(round(ys[select]['lr']))
    tag = model+label_ext+'-ID:'+str(ID)
    return x, y, tag

def plot(ax, x, y, tag):
    ax.plot(x,y, label=tag)
def add_legend(ax):
    ax.legend(frameon=False,bbox_to_anchor =(0.5,-0.3), loc='lower center',ncol=4)


fig, ax = plt.subplots(figsize=(6,6))
x,y,tag = get_results()
plot(ax, x, y, tag)

x,y,tag = get_results(ID=2)
plot(ax, x, y, tag)
add_legend(ax)


plt.show()





