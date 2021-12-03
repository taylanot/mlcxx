import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from models import Bayes, Linear, Ridge, SGD
from tqdm import tqdm
from argparse import Namespace
from sacred import Experiment
from sacred.observers import FileStorageObserver
import copy
import os 
import time

start = time.time()
#ex = Experiment('Marco')


def MSE(y,yp):
    return (yp-y).T.dot((yp-y)) / (y.size)

def sample(dim, m, c, b, Na, Nz, std_y, Ntrn, Ntst):
    a = np.random.multivariate_normal(np.ones(dim)*m,np.eye(dim)*c, Na)
    xtst =  np.random.uniform(0,b,(Nz,Ntst,dim))
    xtrn =  np.random.uniform(0,b,(Nz,Ntrn,dim))
    test_set = []
    train_set = []
    for i in range(Na):
        ytst = (xtst @ a[i].reshape(-1,1)) 
        ytst += np.random.normal(0,std_y,(ytst.shape))
        ytrn = (xtrn @ a[i].reshape(-1,1)) 
        ytrn += np.random.normal(0,std_y,(ytrn.shape))
        test_set.append((xtst, ytst))
        train_set.append((xtrn, ytrn))
    return train_set, test_set

#@ex.config
#def base_config():
#    run_tag = 'std_y' 
#
#    config ={}
#    
#    config['dim'] = 1
#    if run_tag == 'dim':
#        config['dim'] = np.arange(1, 20)
#
#    config['m'] = 5 
#
#    config['c'] = 1
#    if run_tag == 'c':
#        config['c'] = np.linspace(0,10,20)
#
#    config['b'] = 1
#    if run_tag == 'b':
#        config['b'] = np.linspace(1,5,20)
#
#    config['Na'] = 10
#    config['Nz'] = 10
#
#    config['std_y'] = 1
#    if run_tag == 'std_y':
#        config['std_y'] = np.linspace(0,3,20)
#
#    config['single_models'] = ['Bayes','Linear']
#    config['multi_models'] = ['SGD','Ridge']
#    
#    config['lr'] = np.linspace(1e-4, 1,5)
#    if run_tag =='b':
#        config['lr'] = np.linspace(1e-6, 1e-4,5)
#
#    config['alpha'] = np.linspace(1e-2,5,5)
#
#    config['Ntrn'] = 10 
#    if run_tag == 'Ntrn':
#        config['Ntrn'] = np.arange(1, 20)
#    config['Ntst'] = 5000
#    global_vars = copy.deepcopy(config)
#    local_vars = []
#    running_on = "NOTHING!"
#    for key, values in config.items():
#        if  type(values) == np.ndarray and key != 'lr' and key != 'alpha':
#            for value in values:
#                running_on = key
#                global_vars.pop(key)
#                plot_title = copy.deepcopy(global_vars)
#                global_vars.update({key:value})
#                local_vars.append(copy.deepcopy(global_vars))
#    plot_title.pop('single_models')
#    plot_title.pop('multi_models')
#    plot_title.pop('lr')
#    plot_title.pop('alpha')

@numba.jit()
def EE(params):
    var = Namespace(**params)
    trn_sets, tst_sets = sample(dim=var.dim, m=var.m, c=var.c, b=var.b, Na=var.Na,\
            Nz=var.Nz, std_y=var.std_y, Ntrn=var.Ntrn, Ntst=var.Ntst)
    err_a = np.zeros((len(var.single_models)+len(var.lr)+len(var.alpha),var.Na))
    for i in (range(var.Na)):
        err_z = np.zeros((len(var.single_models)+len(var.lr)+len(var.alpha),var.Nz))
        trn_set, tst_set = trn_sets[i], tst_sets[i]
        xtrns, ytrns = trn_set[0], trn_set[1]
        xtsts, ytsts = tst_set[0], tst_set[1]
        for j in range(var.Nz):
            xtrn, ytrn = xtrns[j], ytrns[j]
            xtst, ytst = xtsts[j], ytsts[j]
            # Single-run models 
            for k,model in enumerate(var.single_models):
                model = globals()[model]
                if model == Bayes:
                    model = Bayes(1, var.std_y)
                elif model == Linear:
                    model = Linear()
                model.fit(xtrn, ytrn)
                err_z[k,j] = (MSE(ytst, model.predict(xtst)))
            # Multi-run models
            for model in var.multi_models:
                model = globals()[model]
                if model == SGD:
                    for l, lr in enumerate(var.lr,len(var.single_models)-1):
                        model = SGD(lr=lr)
                        model.fit(xtrn, ytrn)
                        err_z[k+l,j] = (MSE(ytst, model.predict(xtst)))
                elif model == Ridge:
                    for l, alpha in enumerate(var.alpha,len(var.single_models)+len(var.lr)-1):
                        model = Ridge(alpha=alpha)
                        model.fit(xtrn, ytrn)
                        err_z[k+l,j] = (MSE(ytst, model.predict(xtst)))
        err_a[:,i] = np.mean(err_z,axis=1)
    return (np.mean(err_a,axis=1))

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
params['lr'] = np.array([1]) 
params['alpha'] = np.array([1]) 
params['single_models'] = ['Bayes','Linear']
params['multi_models'] = ['SGD','Ridge']


EE(params)
end = time.time()
print(end-start)


#@ex.automain
#def main(local_vars, config, running_on, global_vars, plot_title):
#    if ex.current_run.observers:
#        local_run_path = (ex.current_run.observers[0].dir)
#    if not local_vars:
#        local_vars = [global_vars]
#    print(local_vars)
#    errs = []
#    for var in tqdm(local_vars):
#        errs.append(EE(var).tolist())
#    errs = np.asarray(errs)
#    color = iter(cm.rainbow(np.linspace(0, 1, errs.shape[1])))
#    result = dict()
#    plt.figure(figsize=(8,6))
#    for i, model in enumerate(config['single_models']):
#        c =next(color)
#        plt.plot(config[running_on],errs[:,i], c=c, label=model)
#        result.update({model:errs[:,i].tolist()})
#    for i, model in enumerate(config['multi_models'],2):
#        if model == 'SGD':
#            for j,lr in enumerate(config['lr']):
#                c =next(color)
#                plt.plot(config[running_on],errs[:,i+j],':',c=c,label=model+'-lr:'+str(lr),linewidth=2)
#                result.update({model+'-lr:'+str(lr):errs[:,i+j].tolist()})
#        if model == 'Ridge':
#            for j,alpha in enumerate(config['alpha'],len(config['lr'])-1):
#                c =next(color)
#                plt.plot(config[running_on],errs[:,i+j],'--',c=c,label=model+'-alpha:'+str(alpha))
#                result.update({model+'-alpha:'+str(alpha):errs[:,i+j].tolist()})
#    plt.legend()
#    if 'local_run_path' in locals():
#        filename = os.path.join(local_run_path,str(running_on))
#        plt.title(str(plot_title))
#        plt.xlabel(running_on)
#        plt.ylabel('Expected Error')
#        plt.savefig(filename+'.pdf')
#    else:
#        plt.title(plot_title)
#        plt.xlabel(running_on)
#        plt.ylabel('Expected Error')
#        plt.show()
#    return result
