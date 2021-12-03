import matplotlib.pyplot as plt
import numpy as np
import yaml
import json
import os 

directory = '1D'
exp = '1'
models = ['Bayes', 'MAML', 'Linear', 'Ridge']
models = ['Bayes', 'MAML', 'Ridge']
run_file = 'run.json'

def my_config():
    params = dict()
    params['dim'] = 1
    params['m'] = 0
    params['c'] = np.array([0.,0.5,1.,5.])
    params['std_y'] = np.linspace(0,3,10)
    params['b'] = np.linspace(1,5,5)
    params['Ntrn'] = [1,2,5,10]
    params['Ntst'] = 5000
    params['Nz'] = 100
    params['Na'] = 100 
    params['lr'] = np.linspace(1e-3,1,5)
    params['alpha'] = np.linspace(0,5,5)
    return params

params = my_config()


plt.figure(figsize=(8,6))

def plot_c(j=0,k=0,l=0):
    for model in models:
        run_path = os.path.join(directory,model,exp,run_file)

        with open(run_path) as json_file:
            run = json.load(json_file)
        result =  np.asarray(run['result'])

        if model != 'Ridge' and model != 'MAML':
            plt.plot(params['c'],result[:,j,k,l], label=model)

        elif model == 'Ridge':
            for m in range(len(params['alpha'])):
                plt.plot(params['c'],result[:,j,k,l,m], '--', label=model+'-alpha:{}'.format(params['alpha'][m]))

        elif model == 'MAML':
            for m in range(len(params['lr'])):
                plt.plot(params['c'],result[:,j,k,l,m], ':', label=model+'-lr:{}'.format(params['lr'][m]))

def plot_std_y(i=0,k=0,l=0):
    for model in models:
        run_path = os.path.join(directory,model,exp,run_file)

        with open(run_path) as json_file:
            run = json.load(json_file)
        result =  np.asarray(run['result'])

        if model != 'Ridge' and model != 'MAML':
            plt.plot(params['std_y']**2,result[i,:,k,l], label=model)



        elif model == 'Ridge':
            for m in range(len(params['alpha'])):
                plt.plot(params['std_y']**2,result[i,:,k,l,m],'--', label=model+'-alpha:{}'.format(params['alpha'][m]))

        elif model == 'MAML':
            for m in range(len(params['lr'])):
                plt.plot(params['std_y']**2,result[i,:,k,l,m],':', label=model+'-lr:{}'.format(params['lr'][m]))

def plot_b(i=0,j=0,l=0):
    for model in models:
        run_path = os.path.join(directory,model,exp,run_file)

        with open(run_path) as json_file:
            run = json.load(json_file)
        result =  np.asarray(run['result'])

        if model != 'Ridge' and model != 'MAML':
            plt.plot(params['b'],result[i,j,:,l], label=model)



        elif model == 'Ridge':
            for m in range(len(params['alpha'])):
                plt.plot(params['b'],result[i,j,:,l,m],'--', label=model+'-alpha:{}'.format(params['alpha'][m]))

        elif model == 'MAML':
            for m in range(len(params['lr'])):
                plt.plot(params['b'],result[i,j,:,l,m],':', label=model+'-lr:{}'.format(params['lr'][m]))
                #plt.ylim([0,30])

def plot_Ntrn(i=0,j=0,k=0):
    for model in models:
        run_path = os.path.join(directory,model,exp,run_file)

        with open(run_path) as json_file:
            run = json.load(json_file)
        result =  np.asarray(run['result'])

        if model != 'Ridge' and model != 'MAML':
            plt.plot(params['Ntrn'],result[i,j,k,:], label=model)



        elif model == 'Ridge':
            for m in range(len(params['alpha'])):
                plt.plot(params['Ntrn'],result[i,j,k,:,m], '--', label=model+'-alpha:{}'.format(params['alpha'][m]))

        elif model == 'MAML':
            for m in range(0,len(params['lr'])):
                plt.plot(params['Ntrn'],result[i,j,k,:,m], ':', label=model+'-lr:{}'.format(params['lr'][m]))
                #plt.ylim([0,0.2])


def plot(var='std_y',i=0,j=0,k=0,l=0):
    if var == 'std_y':
        plot_std_y(i,k,l)
        plt.title('c:{}/b:{}/Ntrn:{}'.format(params['c'][i],params['b'][k],params['Ntrn'][l]))
        plt.xlabel('std_y^2')
    if var == 'c':
        plot_c(j,k,l)
        plt.title('std_y:{}/b:{}/Ntrn:{}'.format(params['std_y'][j],params['b'][k],params['Ntrn'][l]))

    if var == 'b':
        plot_b(i,j,l)
        plt.title('c:{}/std_y:{}/Ntrn:{}'.format(params['c'][i],params['std_y'][j],params['Ntrn'][l]))
        plt.xlabel('b')

    if var == 'Ntrn':
        plot_Ntrn(i,j,k)
        plt.title('c:{}/std_y:{}/b:{}'.format(params['c'][i],params['std_y'][j],params['b'][k]))
        plt.xlabel('Ntrn')


plot(var='std_y',i=2,j=0,k=0,l=-1)
plt.legend()
plt.ylabel('Expected Error')
plt.show()


