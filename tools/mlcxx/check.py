import numpy as np
import matplotlib.pyplot as plt
import os

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, learning_curve

def sample_test(a=100, phi=np.pi, N=50, noise=True, sort=True, width=5):
    #x = np.random.uniform(-width, width, (N,1))
    x = np.random.normal(0, 2, (N,1))
    if sort:
        x = np.sort(x,axis=0)
        if noise:
            y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,0.1,(N,1))
        else:
            y = np.multiply(a, np.sin(x+phi)) 
    return x, y 

def load_csv(filename,order="row"):
    if order != "row":
        return np.genfromtxt(filename, delimiter=',').t()
    else:
        return np.genfromtxt(filename, delimiter=',')

def MSE(x,x_):
    assert x.shape == x_.shape
    return ((x-x_)**2).mean()
 
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

class KernelRidge(BaseEstimator):
    def __init__(self, lmbda, kernel):
        self.lmbda= lmbda
        self.kernel = kernel

    def fit(self, X, Y):
        self.fit_them_all(X,Y)
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
                (1e-4+self.lmbda)*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_inter(self, X, Y):
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
                self.lmbda*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_them_all(self, X, Y, grid=100):
        ls = np.linspace(0.001, 100,grid)
        lmbdas = np.linspace(0, 100, grid)
        Xtrn, Xtst, Ytrn, Ytst = train_test_split(X,Y, test_size=0.2)
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
        #print(self.kernel.l)
        #print(self.lmbda)

    def predict(self, X):
        return (self.kernel(X, self.X)).dot(self.alpha)

class SemiParamKernelRidge(BaseEstimator):
    def __init__(self, lmbda, kernel, funcs):
        self.lmbda= lmbda
        self.kernel = kernel
        self.funcs = funcs 

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

    def fit_them_all(self, X, Y, grid=1):
        ls = np.linspace(0.001,100,grid)
        lmbdas = np.linspace(0., 1000, grid)
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
        A = np.block([[K,psi_]])
        B = np.block([[K,np.zeros((X.shape[0],psi_.shape[1]))],[np.zeros((psi_.shape[1],X.shape[0]+psi_.shape[1]))]])
        
        self.w = np.linalg.pinv(A.T.dot(A)+self.lmbda+B.T).dot(A.T.dot(Y))


    def predict(self, X):
        psi_ = self.funcs(X)
        A = np.block([[self.kernel(X, self.X),psi_]])
        return A.dot(self.w)
    def predict2(self, X):
        psi_ = self.funcs(X)
        return psi_.dot(self.w[-len(self.funcs)-1:-1].reshape(-1,1))

class func_gen():
    def __init__(self, num=10):
        self.comp= num
        self.a = 1.
        self.phi = np.random.normal(0,1,num)
        counter = 0 
        if num == 0:
            self.a = 0
            self.phi = 0.

    def __len__(self):
        return self.comp

    def __iter__(self):
        return np.sin

    def __call__(self, x):
        return np.hstack(([np.multiply(self.a, np.sin(x+self.phi))]))

class func_gen2():
    def __init__(self, grid, dataset, num=10):
        self.comp= num
        self.dataset = dataset
        self.grid = grid
        
    def __len__(self):
        return self.comp

    def __call__(self, xs):
        index = []
        for x in xs:
            index.append(np.where(self.grid == x)[0][0])
        return np.array([self.dataset[index,i] for i in range(self.num)]).T

class func_pca():
    def __init__(self, funcdata, comp=1):
        self.funcdata = funcdata
        self.fpca = UFPCA(n_components=comp)
        self.comp = comp
        self.fpca.fit(funcdata)
        self.grid = funcdata.argvals['input_dim_0']
        self.eigenfunctions = self.fpca.eigenfunctions.values
        #plt.plot(self.grid,self.eigenfunctions.T.dot(np.array([7,1.2])))
        plt.plot(self.grid,self.eigenfunctions.T)

    def __len__(self):
        return self.comp

    def __call__(self, xs):
        index = []
        for x in xs:
            index.append(np.where(self.grid == x)[0][0])
        return np.array([self.eigenfunctions[i,index] for i in range(self.comp)]).T

l = 1.
lmbda = 0.
M = 4

filename = "LR-LearningCurves/notune"

file_ids = [ f for f in os.listdir(filename)\
     if os.path.isfile(os.path.join(filename,f)) ]

datas = [load_csv(os.path.join(filename,file_id))\
        for file_id in file_ids]

dataset = []
for data in datas:
    dataset.append(data[:,3])

x = datas[0][:,0].reshape(-1,1)
dataset = np.array(dataset).T

funcdata = DenseFunctionalData({'input_dim_0':x.flatten()}, dataset.T)
funcdata = funcdata.smooth(50,10)

func_id = -20 

x_train = np.arange(5,50,1).reshape(-1,1); y_train=dataset[0:x_train.shape[0],func_id]

x_test = x; y_test = dataset[:,func_id]
#funcs = func_gen2(x, dataset, M)
funcs = func_pca(funcdata, M)
#plt.plot(x_train, funcs(x_train))

kernel = rbf(l=l)

model = SemiParamKernelRidge(lmbda, kernel, funcs)
model_base = KernelRidge(lmbda, kernel)
#
#model.fit(x_train,y_train)
#model_base.fit(x_train,y_train)
model.fit_inter(x_train,y_train)
model_base.fit_inter(x_train,y_train)
#model.w[0:-M]=0
print(model.w)

#plt.scatter(x_train, y_train, label='training')
#plt.plot(x_test,model.predict(x_test), label='meta')
#plt.plot(x_test,model_base.predict(x_test), label='standard')
#plt.plot(x_test, y_test,label='test')
#plt.xlabel('N')
#plt.ylabel('Error')
#plt.ylim([0,5])
#plt.legend()

#plt.plot(x_test,model.predict2(x_test), label='meta')








plt.show()


