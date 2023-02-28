#import numpy as np
#import statsmodels.api as sm
#from sklearn.preprocessing import PolynomialFeatures as poly
#import matplotlib.pyplot as plt
#
#np.random.seed(24)
#def datagen(D=1, N=100,sig=1e-2):
#    x = np.random.normal(0,1,(N,D))
#    beta = np.ones((D,1))
#    return x, ((x**2).dot(beta)) + np.random.normal(0,sig,(N,1))
#
#N = 100
#D = 1 
#sig = 0.001
#
#X, y = datagen(D,N, sig)
#feat = poly(2)
##print(feat.fit_transform(X))
##X = np.hstack((X,np.ones(X.shape)))
#X = feat.fit_transform(X)
#
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#
#ax.scatter(X[:,2],X[:,1],y)
#
#model = sm.OLS(y,X,sig).fit()
#y = model.predict(X)
#ax.scatter(X[:,2],X[:,1],y)
#print(model.summary())
#plt.show()
##print(model.t_test(([0,0,1])))
#
##from patsy import DesignInfo
##use_t = bool_like(use_t, "use_t", strict=True, optional=True)
##names = model.model.data.cov_names
##LC = DesignInfo(names).linear_constraint([1,0])
##print(LC)
##print(model.summary())
#
##print(model.model.normalized_cov_p)
##print(model.f_test([[1,0],[1,0]]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
import tikzplotlib
from sklearn.model_selection import train_test_split
n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

np.random.seed(24)

def MSE(x,x_):
    assert x.shape == x_.shape
    return ((x-x_)**2).mean()
    
def sample_train(Na=1,N=50):
    a = np.random.normal(1, 1, (1,Na))
    phi = np.random.normal(0, 1, (1,Na))
    x = np.random.normal(0, 1, (N,Na))
    y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,0.5,(N,Na))
    y = np.multiply((a*x)**2, np.sin(12*x+phi)) + np.random.normal(0,0.5,(N,Na))
    return x, y , a

def sample_test(a=100, phi=np.pi, N=50, noise=True, sort=True, width=5):
    x = np.random.normal(0, 1, (N,1))
    if sort:
        x = np.sort(x,axis=0)
        if noise:
            y = np.multiply((a*x)**2, np.sin(12*x+phi)) + np.random.normal(0,10.,(N,1))
        else:
            y = np.multiply((a*x)**2, np.sin(12*x+phi)) 
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
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
                self.lmbda*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_inter(self, X, Y):
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
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

class SemiParamKernelRidgePast():
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

        print(self.kernel.l)
        print(self.lmbda)


    def optim(self, X, Y, psi_):
        K = self.kernel(X,X)+1e-8*np.eye(X.shape[0])
        psi_inv  = np.linalg.pinv((psi_.T).dot(psi_))


        #self.alpha = np.dot(np.linalg.inv(K - psi_.dot(psi_inv).dot(psi_.T).dot(K) + (self.lmbda+1e-6)*np.eye(X.shape[0])) \
        #         ,(Y - np.dot(psi_.dot(psi_inv), psi_.T.dot(Y))))

        #self.beta = np.dot(psi_inv, (psi_.T.dot(Y) - psi_.T.dot(K.dot(Y))))

        KK = K.dot(K)
        self.alpha = np.dot(np.linalg.inv(KK + self.lmbda*K - K.dot(psi_.dot(psi_inv.dot(psi_.T.dot(K))))), \
                            (-K.dot(psi_.dot(psi_inv.dot(psi_.T))) + K).dot(Y))

        self.beta = np.dot(psi_inv, psi_.T.dot(Y) - psi_.T.dot(K.dot(self.alpha)))

    def predict(self, X):
        psi_ = self.funcs(X)
        return (self.kernel(X, self.X)).dot(self.alpha) + psi_.dot(self.beta)
        #return (self.kernel(X, self.X)).dot(self.alpha) 
        #return psi_.dot(self.beta)

    def predict2(self, X):
        return (self.kernel(X, self.X)).dot(self.alpha) 

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

    def fit_them_all(self, X, Y, grid=100):
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
        return np.hstack(([np.multiply((self.a*x)**2, np.sin(12*x+self.phi))]))

### Block-2 ####

a =  1. #*np.random.normal(1,1)

lmbda = 0.
l = 0.01 
N = 10
Ntst = 1000
repeat = 100
M = 10

kernel = rbf(l=l)
funcs = func_gen(num=M)

phi =  np.random.normal(0,1)

#noise = True
noise = False

x_train_single, y_train_single = sample_test(a=a, phi=phi, N=N,sort=True,noise=noise)

testsets= []
for i in range(repeat):
    testsets.append(sample_test(a=a, phi=phi, N=Ntst,sort=True,noise=noise))

model = SemiParamKernelRidge(lmbda, kernel, funcs)
model_null = SemiParamKernelRidge(lmbda, kernel, funcs, null=True)
model.fit(x_train_single,y_train_single)
model_null.fit(x_train_single,y_train_single)

err_null = []
err = []

for dataset in testsets:
    err.append(model.error(dataset[0], dataset[1]))
    err_null.append(model_null.error(dataset[0], dataset[1]))

H0 = np.array(err_null)
H1 = np.array(err)

print(H0.mean())
print(H1.mean())

from scipy.stats import wilcoxon, friedmanchisquare
print(wilcoxon(H0, H1))






