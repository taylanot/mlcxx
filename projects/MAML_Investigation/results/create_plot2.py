# File Handling Stuff
import json
import os
from glob import glob
# Stats from statistics import mean, pstdev
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

Experiment_Directory = 'unimodal'
run_file = 'run.json'
config_file = 'config.json'
Ntrns = np.array([1,2,10,50])

def get_results(exp_dir=Experiment_Directory, model='Linear', run_tag='std_y', dim=1, Ntrn=1):
    if run_tag == 'Ntrn':
        ID = 1
    else:
        ID = np.where(Ntrns==Ntrn)[0][0]+1
    if run_tag != 'dim':
        run = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), run_file)
        config = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), config_file)
    else:
        run = os.path.join(exp_dir, 'dim', model, run_tag, str(ID), run_file)
        config = os.path.join(exp_dir,'dim', model, run_tag, str(ID), config_file)

    with open(run) as json_file:
        run_info = json.load(json_file)
    with open(config) as json_file:
        config_info = json.load(json_file)

    x = config_info['config'][run_tag]
    ys = run_info['result']

    if len(ys) == 1:
        y = ys[0]
        label_ext = ''
    else:
        ymeans = [np.mean(res) for res in run_info['result']]
        select = ymeans.index(min(ymeans))
        y = ys[select]
        if model == 'Ridge' or model == 'GeneralRidge':
            label_ext = '-$\lambda$:'+str(round(config_info['config']['alpha'][select],4))
        elif model == 'SGD' or model == 'randomSGD' or model == 'MAML':
            label_ext = '-lr:'+str(round(config_info['config']['lr'][select],4))
    if model == 'SGD':
        model = 'GD'
    if model == 'randomSGD':
        model = 'randomGD'
    tag = model+label_ext
    return x, y, tag

def get_results2(exp_dir=Experiment_Directory, model='Linear', run_tag='std_y', dim=1, Ntrn=1):
    if run_tag == 'Ntrn':
        ID = 1
    else:
        ID = np.where(Ntrns==Ntrn)[0][0]+1
    if run_tag != 'dim':
        run = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), run_file)
        config = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), config_file)
    else:
        run = os.path.join(exp_dir, 'dim', model, run_tag, str(ID), run_file)
        config = os.path.join(exp_dir,'dim', model, run_tag, str(ID), config_file)

    with open(run) as json_file:
        run_info = json.load(json_file)
    with open(config) as json_file:
        config_info = json.load(json_file)

    x = config_info['config'][run_tag]
    ys = run_info['result']

    if len(ys) == 1:
        y = ys[0]
        label_ext = ''
    else:
        ymeans = [res[0] for res in run_info['result']]
        print(ymeans)
        select = ymeans.index(min(ymeans))
        y = ys[select]
        if model == 'Ridge' or model == 'GeneralRidge':
            label_ext = '-$\lambda$:'+str(round(config_info['config']['alpha'][select],4))
        elif model == 'SGD' or model == 'randomSGD' or model == 'MAML':
            label_ext = '-lr:'+str(round(config_info['config']['lr'][select],4))
    if model == 'SGD':
        model = 'GD'
    if model == 'randomSGD':
        model = 'randomGD'
    tag = model+label_ext
    return x, y, tag


def logplot(ax, x, y, tag):
    ax.semilogy(x,y, label=tag)

def plot(ax, x, y, tag):
    if tag != 'Bayes':
        ax.plot(x,y, label=tag)
    else:
        ax.plot(x,y,'--', label=tag)


def add_legend(ax,inside=True):
    if inside:
        ax.legend(frameon=False)
    else:
        ax.legend(frameon=False,bbox_to_anchor =(0.5,-0.3), loc='lower center',ncol=4)

def add_label(ax, run_tag):
    plot_labels = {'std_y':'$\sigma$', 'c':'$c$', 'n_iter': '$n$', 'm':'$m$', 'Ntrn':'$N$', 'b':'$b$', 'dim':'$d$'}
    ax.set_xlabel(plot_labels[run_tag])
    ax.set_ylabel("Expected Error")

#x,y,tag = get_results(model='SGD')
#plot(ax, x, y, tag)

def plot_result(id_, run_tag, dim, Ntrn, log=False, ex=[], save=False, name='test',):
    tex_name = 'tex' 
    pdf_name = 'pdf' 
    png_name = 'png' 
    tex_dir = os.path.join(name,tex_name)
    pdf_dir = os.path.join(name,pdf_name)
    png_dir = os.path.join(name,png_name)
    if not os.path.exists(tex_dir):
        os.makedirs(tex_dir)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    fig, ax = plt.subplots(figsize=(6,6))
    if run_tag != 'n_iter':
        models = ['SGD', 'Linear', 'Ridge', 'Bayes', 'randomSGD', 'MAML','GeneralRidge']
        for model in models:
            if model not in ex:
                x,y,tag = get_results(model=model, run_tag=run_tag,dim=dim, Ntrn=Ntrn)
                if log:
                    logplot(ax, x, y, tag)
                else:
                    plot(ax, x, y, tag)
                add_legend(ax)
                add_label(ax,run_tag)
        if save:
            filename = run_tag+'-'+str(dim)+'-'+str(Ntrn)+'-x-'+str(id_)
            if run_tag == 'dim':
                filename = run_tag+'-'+str(Ntrn)+'-x-'+str(id_)
            tikzplotlib.save(os.path.join(tex_dir, filename)+'.tex')
            plt.savefig(os.path.join(png_dir, filename)+".png")
            plt.savefig(os.path.join(pdf_dir, filename)+".pdf")
        else:
            plt.show()
    else:
        models=['SGD', 'MAML', 'randomSGD']#, 'GeneralRidge', 'Ridge', 'Bayes']
        for model in models:
            if model not in ex:
                x,y,tag = get_results(model=model, run_tag=run_tag,dim=dim, Ntrn=Ntrn)
                x_c,y_c = x,y
                if log:
                    logplot(ax, x, y, tag)
                else:
                    plot(ax, x, y, tag)
                
        x,y,tag = get_results2(model='Ridge', run_tag='b',dim=dim, Ntrn=Ntrn)
        value = 1.
        x = np.array(x)
        absolute_val_array = np.abs(x - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_element = y[smallest_difference_index]
        xp = x_c
        yp = np.ones(len(xp))
        ax.plot(xp, yp*closest_element, label=tag)
        x,y,tag = get_results2(model='GeneralRidge', run_tag='b',dim=dim, Ntrn=Ntrn)
        value = 1.
        x = np.array(x)
        absolute_val_array = np.abs(x - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_element = y[smallest_difference_index-1]
        ax.plot(xp, yp*closest_element, label=tag)
        ax.plot(xp, yp,'--',label='Bayes')
        add_legend(ax)
        add_label(ax,run_tag)
        plt.xlim([0,80])
        
        if save:
            filename = run_tag+'-'+str(dim)+'-'+str(Ntrn)+'-x-'+str(id_)
            if run_tag == 'dim':
                filename = run_tag+'-'+str(Ntrn)+'-x-'+str(id_)
            tikzplotlib.save(os.path.join(tex_dir, filename)+'.tex')
            plt.savefig(os.path.join(pdf_dir, filename)+".pdf")
            plt.savefig(os.path.join(png_dir, filename)+".png")
        else:
            plt.show()


run_tags = ['n_iter']
dims = [1,2,10,50]
ntrns = [1,2,10,50]
#run_tag = 'n_iter'
exs = [[],['Linear']]#,['Linear', 'Ridge', 'randomSGD'],['Linear','SGD'],['Linear','randomSGD']]
#exs = [[]]#,['Linear'],['Linear', 'Ridge', 'randomSGD'],['Linear','randomSGD']]
#ex = []
#ex = ['Linear']
#ex = ['Linear', 'randomSGD', 'MAML', 'Bayes', 'SGD']
#ex = ['Linear','SGD']
#ex = ['Linear','randomSGD']
for dim in dims:
    for Ntrn in ntrns:
        for run_tag in run_tags:
            for id_, ex in enumerate(exs):
                plot_result(id_=id_,run_tag=run_tag, dim=dim, Ntrn=Ntrn, ex=ex, save=True, name='unimodal_plots2')
                plt.close()

#plot_result(id_=0,run_tag='m', dim=1, Ntrn=10, ex=['randomSGD'], save=False, name='test')
#plot_result(run_tag, dim=1, Ntrn=2)

### NEED to write some other function for n_iter






