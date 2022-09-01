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

Experiment_Directory = 'unimodal-nonlinear-fix-final'
run_file = 'run.json'
config_file = 'config.json'
Ntrns = np.array([1,2,10,50])

def get_results(exp_dir=Experiment_Directory, model='Linear', run_tag='std_y', dim=1, Ntrn=1):
    if run_tag == 'Ntrn':
        ID = 1
    else:
        ID = np.where(Ntrns==Ntrn)[0][0]+1

    result_file = 'result-'+str(ID)+'-'+str(model)+'.npy'
    if model=='MAML' or model=='GD':
        result_file = 'result-'+str(ID)+'-'+str(model)+str(run_tag)+'.npy'
    if run_tag != 'dim':
        run = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), run_file)
        config = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), config_file)
        res = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), result_file)
    else:
        run = os.path.join(exp_dir, 'dim', model, run_tag, str(ID), run_file)
        config = os.path.join(exp_dir,'dim', model, run_tag, str(ID), config_file)
        res = os.path.join(exp_dir, str(dim), model, run_tag, str(ID), result_file)

    with open(run) as json_file:
        run_info = json.load(json_file)
    with open(config) as json_file:
        config_info = json.load(json_file)

    x = config_info['config'][run_tag]
    #ys = (np.load(res).reshape(-1,20)).tolist()
    print(np.load(res).shape)
    if run_tag == 'Ntrn':
        ys = np.load(res)[0,:].reshape(-1,49)
    elif run_tag == 'n_iter' :
        ys = np.load(res)[0,:].reshape(-1,10)
    else:
        ys = np.load(res)[0,:].reshape(-1,20)
    #ys = np.load(res)

    if len(ys) == 1:
        y = ys[0]
        label_ext = ''
    else:
        ymeans = [np.mean(res) for res in ys]
        select = ymeans.index(min(ymeans))
        y = ys[select]
        if model == 'KernelRidge':
            label_ext = '-$\lambda$:'+str(round(config_info['config']['alpha'][select],4))
        elif model == 'MAML' or model=='GD':
            label_ext = '-lr:'+str(round(config_info['config']['lr'][select],4))
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
    plot_labels = {'std_y':'$\sigma$', 'c_phase':'$c_{2}$', 'n_iter': '$n_{iter}$', 'c_amplitude':'$c_{1}$', 'Ntrn':'$N$'}
    ax.set_xlabel(plot_labels[run_tag])
    ax.set_ylabel("Expected Error")

#x,y,tag = get_results(model='SGD')
#plot(ax, x, y, tag)

def plot_result(id_, run_tag, dim, Ntrn, log=False, ex=[], save=False, name='test',):
    tex_name = 'tikz' 
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
        models = ['KernelRidge', 'Bayes', 'MAML', 'GD']
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
            tikzplotlib.save(os.path.join(tex_dir, filename)+'.tikz')
            plt.savefig(os.path.join(png_dir, filename)+".png")
            plt.savefig(os.path.join(pdf_dir, filename)+".pdf")
        else:
            plt.show()
    else:
        models=['KernelRidge', 'Bayes', 'MAML', 'GD']
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
            tikzplotlib.save(os.path.join(tex_dir, filename)+'.tikz')
            plt.savefig(os.path.join(pdf_dir, filename)+".pdf")
            plt.savefig(os.path.join(png_dir, filename)+".png")
        else:
            plt.show()


run_tags = ['std_y', 'c_phase', 'c_amplitude', 'n_iter', 'Ntrn']
dims = [1,2,10,50]
ntrns = [1,2,10,50]
#run_tag = 'n_iter'
#exs = [[],['Linear'],['Linear', 'randomSGD']]#,['Linear', 'Ridge', 'randomSGD'],['Linear','SGD'],['Linear','randomSGD']]
exs = [[]]#,['Linear'],['Linear', 'randomSGD']]#,['Linear', 'Ridge', 'randomSGD'],['Linear','SGD'],['Linear','randomSGD']]
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
                plot_result(id_=id_,run_tag=run_tag, dim=dim, Ntrn=Ntrn, ex=ex, save=True, name='unimodal_plots_nonlinear')
                plt.close()

#plot_result(id_=0,run_tag='Ntrn', dim=1, Ntrn=10, ex=exs, save=False, name='test')
#plt.show()
#plot_result(run_tag, dim=1, Ntrn=2)

### NEED to write some other function for n_iter






