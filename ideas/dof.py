import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
import tikzplotlib
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from copy import deepcopy
from functional_pca import *
from scipy.stats import wilcoxon, friedmanchisquare, ttest_ind, f_oneway,f
from statistics import mean

n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

np.random.seed(20)

def MSE(x,x_):
    assert x.shape == x_.shape
    return ((x-x_)**2).mean()
    
def sample(a=100, phi=np.pi, N=50, noise=True, sort=True, width=5,test=False):
    if test:
        x = np.random.normal(0, 5, (N,1))
    else:
        x = np.random.normal(0, 1, (N,1))
    if sort:
        x = np.sort(x,axis=0)
        if noise:
            #y = np.multiply((a*x)**2, np.sin(12*x+phi)) + np.random.normal(0,1.,(N,1))
            y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,.1,(N,1))
        else:
            #y = np.multiply((a*x)**2, np.sin(12*x+phi)) 
            y = np.multiply(a, np.sin(x+phi)) 
    return x, y 


def pairwise_l2_distance(x,y):
    D = -2 * x @  y.T + np.sum(y**2, axis=1) + \
            np.sum(x**2, axis=1)[:,np.newaxis]
    D[D<0] = 0.
    return D


class rbf():
    def __init__(self,l=1):
        self.l = l

    def __call__(self, x, xp=None):
        if np.any(xp == None):
            xp = x
        alpha = pairwise_l2_distance(x,xp)
        return np.exp(-alpha/self.l**2)


class KernelRidge():
    def __init__(self, lmbda, kernel):
        self.lmbda= lmbda
        self.kernel = kernel

    def fit(self, X, Y):
        self.fit_them_all(X,Y)
        self.alpha = np.linalg.pinv(self.kernel(X,X) + \
                (self.lmbda)*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_inter(self, X, Y):
        self.alpha = np.linalg.pinv(self.kernel(X,X) + \
                (self.lmbda)*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_them_all(self, X, Y, grid=50):
        ls = np.linspace(0.1,1,grid)
        lmbdas = np.linspace(0, 1, grid)
        Xtrn, Xtst, Ytrn, Ytst = train_test_split(X,Y, test_size=0.2, random_state=2)
        error = np.zeros((grid, grid))
        for i, l in enumerate(ls):
            self.kernel.l = l 
            for j, lmbda in enumerate(lmbdas):
                self.lmbda = lmbda
                self.fit_inter(Xtrn, Ytrn)
                error[i, j] = MSE(Ytst, self.predict(Xtst))

        index = np.where(error == np.amin(error))
        self.kernel.l = ls[index[0][0]]
        self.lmbda = lmbdas[index[1][0]]

    def predict(self, X):
        return (self.kernel(X, self.X)).dot(self.alpha)
    def error(self, X, y):
        return MSE(self.predict(X), y)
    def SS(self, X, y):
        return self.error(X,y)*self.X.shape[0]





class SemiParamKernelRidge():
    def __init__(self, lmbda, kernel, funcs, null=False):
        self.lmbda= lmbda
        self.kernel = kernel
        self.funcs = funcs 
        self.null = null

    def fit(self, X, Y):
        self.fit_them_all(X,Y)
        self.X = X
        self.Y = Y
        self.psi_ = self.funcs(X)
        self.optim(X,Y,self.psi_)

    def fit_inter(self, X, Y):
        self.psi_ = self.funcs(X)
        self.optim(X,Y,self.psi_)
        self.Y = Y
        self.X = X

    def fit_them_all(self, X, Y, grid=50):
        ls = np.linspace(0.001,1,grid)
        lmbdas = np.linspace(0., 10, grid)
        Xtrn, Xtst, Ytrn, Ytst = train_test_split(X,Y, test_size=0.5, random_state=2)
        error = np.zeros((grid, grid))
        for i, l in enumerate(ls):
            self.kernel.l = l 
            for j, lmbda in enumerate(lmbdas):
                self.lmbda = lmbda
                self.fit_inter(Xtrn, Ytrn)
                error[i, j] = MSE(Ytst, self.predict(Xtst))
        index = np.where(error == np.amin(error))
        self.kernel.l = ls[index[0][0]]
        self.lmbda = lmbdas[index[1][0]]


    def optim(self, X, Y, psi_):
        K = self.kernel(X,X)
        if self.null:
            K = np.zeros((X.shape[0], X.shape[0]))
        A = np.block([[K,psi_]])
        B = np.block([[K,np.zeros((X.shape[0],psi_.shape[1]))],[np.zeros((psi_.shape[1],X.shape[0]+psi_.shape[1]))]])
        
        self.w = np.linalg.pinv(A.T.dot(A)+self.lmbda*B.T).dot(A.T.dot(Y))
        #print(self.w)
        self.alpha = self.w[0:X.shape[0]]


    def predict(self, X):
        psi_ = self.funcs(X)
        A = np.block([[self.kernel(X, self.X),psi_]])
        return A.dot(self.w)
        #return (self.kernel(X, self.X)).dot(self.alpha) + psi_.dot(self.beta)
        #return (self.kernel(X, self.X)).dot(self.alpha) 
        #return psi_.dot(self.beta)
    def error(self, X, y):
        return MSE(self.predict(X), y)
    def SS(self, X, y):
        return self.error(X,y)*self.X.shape[0]



class func_gen_pca():
    def __init__(self, num=1000):
        self.num = num
        self.a = np.random.normal(1,0.1,num)
        self.phis = np.random.normal(0,1,num)
        counter = 0 
        if num == 0:
            self.a = 0
            self.phi = 0.

    def __len__(self):
        return self.num

    def __iter__(self):
        return np.sin

    def __call__(self, x):
        N = x.shape[0]
        M = 2
        x = x.flatten()
        X = np.sort(x); Y = np.array([np.sin(x+phi) for phi in self.phis])
        _, eigenfunctions, _ = eig(X,Y,False)

        return eigenfunctions.T

class func_gen():
    def __init__(self, num=10):
        self.num = num
        self.a = 1.#np.random.normal(1,0.1,num)
        self.phi = np.random.normal(0,1,num)
        counter = 0 
        if num == 0:
            self.a = 0
            self.phi = 0.

    def __len__(self):
        return self.num

    def __iter__(self):
        return np.sin

    def __call__(self, x):
        #return np.hstack(([np.multiply(self.a, np.sin(x+self.phi))]))
        return np.hstack(([np.multiply(self.a, 1./(1.+np.exp(-x)))]))

a =  1
lmbda = 0.1
l = 1. 
N = 10
Ntst = 1000
M = 10
reboot = 100
repeat= 1
kernel = rbf(l=l)
phi =  np.random.normal(0,1)
#phi =  funcs.phi[0]

funcs = func_gen(num=M)
noise = True

model0 = SemiParamKernelRidge(lmbda, kernel,funcs,True)
model1 = SemiParamKernelRidge(lmbda, kernel,funcs)
model2 = KernelRidge(lmbda, kernel)

models = [model0, model1, model2]

def get_df(model):
    dfs = []
    for repeat in range(100):
        x, y = sample(a=a, phi=phi, N=N,sort=True,noise=noise)
        preds = []
        for i in range(reboot):
            xi, yi = resample(x, y)
            model.fit_inter(xi,yi)
            preds.append(model.predict(x))
        dfs.append(1./0.1*np.trace(np.cov(np.array(preds).reshape(N,-1))))
    return sum(dfs)/len(dfs)

df = [get_df(model) for model in models]

f01s, f02s, f12s = list(), list(), list()
for i in range(repeat):
    x, y = sample(a=a, phi=phi, N=N,sort=True,noise=noise)
    dataset = sample(a=a, phi=phi, N=Ntst,sort=True,noise=noise,test=True)
    trained = [model.fit(x,y) for model in models]
    SS = [model.SS(dataset[0],dataset[1]) for model in models]
    F01 = ((SS[0]-SS[1])/(df[0]-df[1]))/(SS[1]/df[1])
    F02 = ((SS[0]-SS[2])/(df[0]-df[2]))/(SS[2]/df[2])
    F12 = ((SS[1]-SS[2])/(df[1]-df[2]))/(SS[2]/df[2])
    f01s.append(f.cdf(F01, df[0], df[1]))
    f02s.append(f.cdf(F02, df[0], df[2]))
    f12s.append(f.cdf(F12, df[1], df[2]))

print('f01:',mean(f01s))
print('f02:',mean(f02s))
print('f12:',mean(f12s))




