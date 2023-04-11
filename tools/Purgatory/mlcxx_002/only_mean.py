import numpy as np
import matplotlib.pyplot as plt
import os

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, learning_curve

def integration_weights_(x):
    return 0.5 * np.concatenate((np.array([x[1] - x[0]]),
                              x[2:] - x[:(len(x) - 2)],
                              np.array([x[len(x) - 1] - x[len(x) - 2]]))
                              ,axis=None)
def eig(X,Y, mean_sub=True, npc=1):
    n_obs = X.size
    argvals = X.copy()
    data = Y.copy()
    weight = integration_weights_(argvals)
    #print(weight)
    # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
    weight_sqrt = np.diag(np.sqrt(weight))
    weight_invsqrt = np.diag(1 / np.sqrt(weight))
    mean = Y.mean(axis=0)
    if mean_sub:
        data -= mean
    covariance = data.T.dot(data) / (n_obs-1)
    covariance += covariance.T
    covariance /= 2
    var = np.dot(np.dot(weight_sqrt, covariance), weight_sqrt)

    eigenvalues, eigenvectors = np.linalg.eigh(var)
    eigenvalues[eigenvalues < 0] = 0
    eigenvalues = eigenvalues[::-1]
    #print(eigenvalues)
    # Slice eigenvalues and compute eigenfunctions = W^{-1/2}U
    #exp_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    #print(exp_variance)
    #npc = np.sum(exp_variance < 0.99) + 1

    eigenvalues = eigenvalues[:npc]
    eigenfunctions = np.transpose(np.dot(weight_invsqrt,
                                         np.fliplr(eigenvectors)[:, :npc]))
    # (NxN @ Nxnpc)^T -> npcxN

    return eigenvalues, eigenfunctions, npc

def eig_alt(X,Y, mean_sub=False, npc=1):
    n_obs = X.size
    argvals = X.copy()
    data = Y.copy()
    weight = integration_weights_(argvals)
    #print(weight)
    # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
    weight_sqrt = np.diag(np.sqrt(weight))
    weight_invsqrt = np.diag(1 / np.sqrt(weight))
    mean = Y.mean(axis=0)
    if mean_sub:
        data -= mean
    covariance = data.T.dot(data) / (n_obs-1)
    covariance += covariance.T
    covariance /= 2
    var = np.dot(np.dot(weight_sqrt, covariance), weight_sqrt)

    eigenvalues, eigenvectors = np.linalg.eigh(var)
    eigenvalues[eigenvalues < 0] = 0
    eigenvalues = eigenvalues[::-1]
    #print(eigenvalues)
    # Slice eigenvalues and compute eigenfunctions = W^{-1/2}U
    #exp_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    #print(exp_variance)
    #npc = np.sum(exp_variance < 0.99) + 1

    eigenvalues = eigenvalues[:npc]
    eigenfunctions = np.transpose(np.dot(weight_invsqrt,
                                         np.fliplr(eigenvectors)[:, :npc]))
    # (NxN @ Nxnpc)^T -> npcxN

    return eigenvalues, eigenfunctions+np.tile(np.mean(funcdata.values,axis=0),(npc,1)), npc

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

class func_pca():
    def __init__(self, funcdata, comp=1, mean_sub=False):
        self.funcdata = funcdata
        self.fpca = UFPCA(n_components=comp)
        self.comp = comp
        self.fpca.fit(funcdata)
        self.grid = funcdata.argvals['input_dim_0']
        #self.eigenfunctions = self.fpca.eigenfunctions.values
        _,self.eigenfunctions,_ = eig_alt(self.grid, funcdata.values,npc=comp)#,mean_sub=mean_sub)
        self.eigenfunctions += np.mean(funcdata.values, axis=0)

    def __call__(self, xs):
        index = []
        for x in xs:
            index.append(np.where(self.grid == x)[0][0])
        #return np.array([self.eigenfunctions[i, index] for i in range(self.comp)]).reshape(-1,self.comp)
        return np.array([self.eigenfunctions[i,index] for i in range(self.comp)]).T

class func_mean():
    def __init__(self, funcdata, comp=1, mean_sub=False):
        self.funcdata = funcdata
        self.fpca = UFPCA(n_components=comp)
        self.comp = comp
        self.fpca.fit(funcdata)
        self.grid = funcdata.argvals['input_dim_0']
        #self.eigenfunctions = self.fpca.eigenfunctions.values
        self.eigenfunctions = np.mean(funcdata.values, axis=0)
    def __call__(self, xs):
        index = []
        for x in xs:
            index.append(np.where(self.grid == x)[0][0])
        #return np.array([self.eigenfunctions[i, index] for i in range(self.comp)]).reshape(-1,self.comp)
        return self.eigenfunctions[index].reshape(-1,1)

class func_raw():
    def __init__(self, funcdata, comp=1, mean_sub=False):
        self.funcdata = funcdata
        self.fpca = UFPCA(n_components=comp)
        self.comp = comp
        self.fpca.fit(funcdata)
        self.grid = funcdata.argvals['input_dim_0']
        self.eigenfunctions = self.funcdata.values
    def __call__(self, xs):
        index = []
        for x in xs:
            index.append(np.where(self.grid == x)[0][0])
        return np.array([self.eigenfunctions[i, index] for i in range(self.comp)]).reshape(-1,self.comp)

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
        #print(self.kernel.l)
        #print(self.lmbda)


    def optim(self, X, Y, psi_):
        K = self.kernel(X,X)
        A = np.block([[K,psi_]])
        B = np.block([[K,np.zeros((X.shape[0],psi_.shape[1]))],[np.zeros((psi_.shape[1],X.shape[0]+psi_.shape[1]))]])
        self.w = np.linalg.pinv(A.T.dot(A)+self.lmbda*B.T).dot(A.T.dot(Y))
        self.beta = self.w[X.shape[0]:]


    def predict(self, X):
        psi_ = self.funcs(X)
        A = np.block([[self.kernel(X, self.X),psi_]])
        return A.dot(self.w)
    def predict2(self, X):
        psi_ = self.funcs(X)
        return psi_.dot(self.w[-len(self.funcs)-1:-1].reshape(-1,1))

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

M = 60
func_id = -1
Ntrn = 60
l = 1
lmbda = 0

filename = "LearningCurves/1D/LR-LearningCurves/notune"

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
#funcdata = funcdata.smooth(10,10)
#funcdata = funcdata.smooth(50,10)

Y = funcdata.values
X = funcdata.argvals['input_dim_0']

x_train = np.arange(5,Ntrn,1).reshape(-1,1); y_train=dataset[0:x_train.shape[0],func_id]
x_test = x; y_test = dataset[:,func_id]

#funcs = func_mean(funcdata,comp=M, mean_sub=False)
funcs = func_pca(funcdata, M)
#funcs = func_raw(funcdata, M)
#plot(funcdata)
#plt.show()

kernel = rbf(l=l)
model = SemiParamKernelRidge(lmbda, kernel, funcs)
model.fit_inter(x_train,y_train)
#print(np.linalg.norm(model.beta,2))
print(model.beta)
#model.beta *= 0.5
plt.scatter(x_train, y_train, label='training')
plt.plot(x_test,model.predict(x_test), label='meta')
plt.plot(x_test, y_test,label='test')
plt.xlabel('N')
plt.ylabel('Error')
plt.legend()
plt.show()

#funcy = func_pca(funcdata,comp=M, mean_sub=True)
#
#funcs = funcy(X)
##_, funcs, _ = eig(X,Y,npc=M)
#plt.plot(X,funcs, label="mean")
#
#funcy = func_pca(funcdata,comp=M, mean_sub=False)
#
#funcs = funcy(X)
##_, funcs, _ = eig(X,Y,npc=M)
#plt.plot(X,funcs+np.tile(np.mean(funcdata.values,axis=0),(3,1)).T, label="nomean")
#plt.legend()


