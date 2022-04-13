import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean, stdev
from sklearn.datasets import make_friedman2
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels  import RBF, Matern, ExpSineSquared

def f(x, noise=False):
    assert x.ndim == 1.
    y = x*np.sin(x)
    if noise:
        rnd = 0.5 + np.random.random(y.shape)
        eps  = np.random.normal(0,rnd)
        return  (y + eps).ravel()
    else:
        return y.ravel()

def create_datasets(N=40, seed=24):
    np.random.seed(seed)
    x = np.random.uniform(0,10,(N))
    y_noiseless = f(x,noise=False)
    y_noisy = f(x,noise=True)
    sort = np.argsort(x)
    noiseless_data = pd.DataFrame(data={'x':x[sort], 'y':y_noiseless[sort]})
    noisy_data = pd.DataFrame(data={'x':x[sort], 'y':y_noisy[sort]})
    return noiseless_data, noisy_data


def split(dataset, size=0.2, state=81):
    Xtrn, Xtst, ytrn, ytst = train_test_split(dataset['x'].to_numpy().reshape(-1,1),\
            dataset['y'].to_numpy(), test_size=size, random_state=state)
    train, test  = (Xtrn, ytrn), (Xtst, ytst)
    return train, test


def experiment(model, dataset, Ntrn=10, xval=False):
    train, test = split(dataset)
    if xval:
        kf = KFold(n_splits=6)
        Xtrns, ytrns = train
        r2s, mses = [], []
        for itrn, itst in kf.split(Xtrns):
            Xtrn, ytrn = Xtrns[itrn], ytrns[itrn]
            Xtst, ytst = Xtrns[itst], ytrns[itst]
            model.fit(Xtrn, ytrn)
            ypred = model.predict(Xtst)
            r2s.append(r2_score(ytst, ypred))
            mses.append(mean_squared_error(ytst, ypred))
        return [(mean(r2s), stdev(r2s)),(mean(mses), stdev(mses))]
            
    else:
        Xtrn, ytrn = train
        Xtst, ytst = test
        print(Xtrn[0:Ntrn],ytrn[0:Ntrn])
        model.fit(Xtrn[0:Ntrn],ytrn[0:Ntrn])
        ypred = model.predict(Xtst)
        return [r2_score(ytst, ypred), mean_squared_error(ytst, ypred)]

    
def experiment_1(degrees=[1,3,5,10,20], Ntrns=[6,11,21]):
    datasets = create_datasets()
    for i, dataset in enumerate(datasets): 
        if i == 0:
            print('noiseless_dataset:')
        elif i == 1:
            print('noisy_dataset:')
        for degree in degrees:
            for Ntrn in Ntrns:
                model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
                                ('linear', LinearRegression(fit_intercept=False))])
                err = experiment(model, dataset, Ntrn)
                print('degree:{}/Ntrn:{}'.format(degree, Ntrn))
                print('r2:{}/mse:{}'.format(err[0],err[1]))

def experiment_2(degrees=[1,5,20]):
    datasets = create_datasets()
    for i, dataset in enumerate(datasets): 
        if i == 0:
            print('noiseless_dataset:')
        elif i == 1:
            print('noisy_dataset:')
        for degree in degrees:
            model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
                            ('linear', LinearRegression(fit_intercept=False))])
            err = experiment(model, dataset, xval=True)
            print('degree:{}'.format(degree))
            print('r2-mean:{}/mse-mean:{}'.format(err[0][0],err[1][0]))
            print('r2-std:{}/mse-std:{}'.format(err[0][1],err[1][1]))

def experiment_3(alpha=1.e2, degrees=[1,3,5,10,20], Ntrns=[6,11,21]):
    dataset,_ = create_datasets()
    print('noiseless_dataset:')
    for degree in degrees:
        for Ntrn in Ntrns:
            model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
                            ('ridge', Ridge(alpha=alpha,fit_intercept=False))])
            err = experiment(model, dataset, Ntrn)
            print('degree:{}/Ntrn:{}'.format(degree, Ntrn))
            print('r2:{}/mse:{}'.format(err[0],err[1]))

def experiment_4(kernels=[RBF(), Matern(nu=1.5), Matern(nu=2.5), ExpSineSquared()], Ntrns=[6,11,21]):
    datasets = create_datasets()
    for i, dataset in enumerate(datasets): 
        if i == 0:
            print('noiseless_dataset:')
        elif i == 1:
            print('noisy_dataset:')
        for kern in kernels:
            for Ntrn in Ntrns:
                model = GPR(kernel=kern, n_restarts_optimizer=10)
                err = experiment(model, dataset, Ntrn)
                print('Ntrn:{}'.format(Ntrn))
                print('kernel_info:', model.kernel_)
                print('r2:{}/mse:{}'.format(err[0],err[1]))

def experiment_bonus_1(degrees=[1,3,5,10,20], Ntrns=[6,11,21]):
    dataset,_ = create_datasets()
    train, test = split(dataset)
    Xtrn, ytrn = train
    Xtst, ytst = test
    print('noiseless_dataset:')
    for degree in degrees:
        for Ntrn in Ntrns:
            model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
                            ('ridge', Ridge(alpha=1,fit_intercept=False))])
            grid= GridSearchCV(estimator=model, param_grid={'ridge__alpha':[1e-6,10]}) 
            grid.fit(Xtrn, ytrn)
            print('degree:{}/Ntrn:{}'.format(degree, Ntrn))
            ypred = grid.predict(Xtst)
            err = [r2_score(ytst, ypred), mean_squared_error(ytst, ypred)]
            print('r2:{}/mse:{}'.format(err[0],err[1]))

def experiment_bonus_2(models=[BayesianRidge()], degrees=[1,3,5,10,20], Ntrns=[6,11,21]):
    datasets = create_datasets()
    for i, dataset in enumerate(datasets): 
        if i == 0:
            print('noiseless_dataset:')
        elif i == 1:
            print('noisy_dataset:')
        for degree in degrees:
            for model_type in models:
                name = type(model_type).__name__
                for Ntrn in Ntrns:
                    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
                                (str(name), model_type)])
                    err = experiment(model, dataset, Ntrn)
                    print('Ntrn:{}'.format(Ntrn))
                    print('model_name:',name )
                    print('r2:{}/mse:{}'.format(err[0],err[1]))


#experiment_1()
#experiment_2()
#experiment_3()
#experiment_4()
#experiment_bonus_1()
experiment_bonus_2()
#X, y = make_friedman2(500)
#print(y)

#print(split(D_noisy)[0][0])
#print(split(D_noiseless))
#plt.plot(D_noiseless['x'], D_noiseless['y'])
#plt.show()
