import numpy as np
import matplotlib.pyplot as plt
import os

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, learning_curve


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
                self.lmbda*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_inter(self, X, Y):
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
                self.lmbda*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_them_all(self, X, Y, grid=10):
        ls = np.linspace(self.kernel.l-0.5, self.kernel.l+0.9,grid)
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

    def fit(self, X, Y):
        self.psi_ = self.funcs(X)
        self.optim(X,Y,self.psi_)
        self.Y = Y
        self.X = X
        print(self.kernel.l)
        print(self.lmbda)

    def fit_them_all(self, X, Y, grid=1):
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
        A = np.block([[K,psi_]])
        B = np.block([[K,np.zeros((X.shape[0],psi_.shape[1]))],[np.zeros((psi_.shape[1],X.shape[0]+psi_.shape[1]))]])
        
        #psi_inv  = np.linalg.inv((psi_.T).dot(psi_)+(1e)*np.eye(psi_.shape[1]))

        #self.alpha = np.dot(np.linalg.inv(K - psi_.dot(psi_inv).dot(psi_.T).dot(K) + (self.lmbda+1e-6)*np.eye(X.shape[0])) \
        #         ,(Y - np.dot(psi_.dot(psi_inv), psi_.T.dot(Y))))

        #self.beta = np.dot(psi_inv, (psi_.T.dot(Y) - psi_.T.dot(K.dot(Y))))

        #KK = K.dot(K)
        #self.alpha = np.dot(np.linalg.inv(KK + self.lmbda*K - K.dot(psi_.dot(psi_inv.dot(psi_.T.dot(K))))), \
        #                    (K.dot(psi_.dot(psi_inv.dot(psi_.T))) - K).dot(Y))

        #self.beta = np.dot(psi_inv, psi_.T.dot(Y)- psi_.T.dot(K.dot(self.alpha)))
        self.w = np.linalg.pinv(A.T.dot(A)+self.lmbda+B.T).dot(A.T.dot(Y))


    def predict(self, X):
        psi_ = self.funcs(X)
        
        A = np.block([[self.kernel(X, self.X),psi_]])
        return A.dot(self.w)

class func_pca():
    def __init__(self, funcdata, comp=1):
        self.funcdata = funcdata
        self.fpca = UFPCA(n_components=comp)
        self.comp = comp
        self.fpca.fit(funcdata)
        self.grid = funcdata.argvals['input_dim_0']
        self.eigenfunctions = self.fpca.eigenfunctions.values
    def __call__(self, xs):
        index = []
        for x in xs:
            index.append(np.where(self.grid == x)[0][0])
        return np.array([self.eigenfunctions[i, index] for i in range(self.comp)]).reshape(-1,self.comp)
        

filename = "LR-LearningCurves/notune"

file_ids = [ f for f in os.listdir(filename)\
             if os.path.isfile(os.path.join(filename,f)) ]

datas = [load_csv(os.path.join(filename,file_id))\
                for file_id in file_ids]

train = []
test = []

for data in datas:
    train.append(data[:,1])
    test.append(data[:,3])

train = np.array(train)
test = np.array(test)
grid = datas[0][:,0]
funcdata = DenseFunctionalData({'input_dim_0':grid}, test)
funcdata = funcdata.smooth(50,10)

#data_smooth = funcdata.smooth(points=20, neighborhood=10)
# Perform a univariate FPCA on dailyTemp.
# Perform a univariate FPCA on dailyTemp.
fpca = UFPCA(n_components=0.99)
fpca.fit(funcdata)
_ = plot(fpca.eigenfunctions)
plt.xlabel('N')
plt.ylabel('Error')
plt.show()
#print(fpca.eigenfunctions.values)
#eig_func_vals = []

#l = 0.1
#lmbda = 0.
#
#Xtrn, Ytrn = grid[0:30].reshape(-1,1), test[0,0:30].reshape(-1,1)
#print(Xtrn,Ytrn)
#Xtst, Ytst = grid.reshape(-1,1), test[0].reshape(-1,1)
#
#kernel = rbf(l=l)
#funcs = func_pca(funcdata,1)
#
#model = SemiParamKernelRidge(lmbda, kernel, funcs)
#model.fit(Xtrn, Ytrn)
#plt.plot(Xtst, (model.predict(Xtst)))
#plt.plot(Xtst, Ytst)
#plt.show()

