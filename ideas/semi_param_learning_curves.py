import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import cycler
#import tikzplotlib
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.base import BaseEstimator
#n = 8
#color = plt.cm.Dark2(np.linspace(0, 1,n))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

np.random.seed(24)

def MSE(x,x_):
    assert x.shape == x_.shape
    return ((x-x_)**2).mean()
    
def sample_train(Na=1,N=50):
    a = np.random.normal(1, 1, (1,Na))
    phi = np.random.normal(0, 1, (1,Na))
    x = np.random.normal(0, 1, (N,Na))
    y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,1,(N,Na))
    return x, y , a

def sample_test(a=100,phi=np.pi,N=50):
    x = np.random.normal(0,10, N).reshape(-1,1)
    x = np.sort(x,axis=0)
    y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,1,(N,1))
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

class SemiParamKernelRidge2(BaseEstimator):
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

    def fit_them_all(self, X, Y, grid=10):
        ls = np.linspace(self.kernel.l-0.9, self.kernel.l+0.9,grid)
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


    def optim(self, X, Y, psi_, itr=1000):
        self.alpha = np.zeros(X.shape[0]).reshape(-1,1)
        self.beta = np.zeros(len(self.funcs)).reshape(-1,1)
        K = self.kernel(X,X)
        for i in range(itr):
            self.alpha = np.linalg.pinv( K + self.lmbda*np.eye(X.shape[0])).\
                    dot(Y-psi_.dot(self.beta))
            self.beta = np.linalg.pinv(psi_).dot(Y-K.dot(self.alpha))
            #print(self.alpha)
            #print(self.beta)

    def predict(self, X):
        psi_ = self.funcs(X)
        return (self.kernel(X, self.X)).dot(self.alpha) + psi_.dot(self.beta)

class func_gen():
    def __init__(self, num=10):
        self.num = num
        self.a = np.random.normal(1,1,num)
        self.phi = np.random.normal(1,1,num)
        counter = 0 

    def __len__(self):
        return self.num

    def __iter__(self):
        return np.sin

    def __call__(self, x):
        return np.hstack(([np.multiply(self.a, np.sin(x+self.phi))]))


### Block-2 ####
a =  np.random.normal(1,1)
phi =  np.random.normal(0,1)
x_train_single, y_train_single = sample_test(a=a,phi=phi, N=1000)
kernel = rbf()
model_base = KernelRidge(1, kernel)
for num in range(2, 24,4):
    funcs = func_gen(num=num)
    model = SemiParamKernelRidge2(1., kernel, funcs)

    #model.fit(x_train_single,y_train_single)
    #model_base.fit(x_train_single,y_train_single)
    def plot_learning_curve(estimators):
        ax = [plt.subplot(211), plt.subplot(212)]
        for i,estimator in enumerate(estimators):
            train_size, train_scores, test_scores = \
            learning_curve(estimator, x_train_single, y_train_single,\
            train_sizes=np.linspace(0.01,0.4,5), scoring='neg_mean_squared_error')

            train_scores_mean = np.mean(-train_scores, axis=1)
            train_scores_std = np.std(-train_scores, axis=1)
            test_scores_mean = np.mean(-test_scores, axis=1)
            test_scores_std = np.std(-test_scores, axis=1)
            
            ax[i].plot(train_size, train_scores_mean, label='train_mean')
            ax[i].plot(train_size, test_scores_mean, label='test_mean')
        plt.legend()

    plt.figure()
    plot_learning_curve([model_base, model])
    plt.savefig('learning'+str(num)+'.pdf')
plt.show()
#plt.scatter(x_train_single, y_train_single)
#x_train_single, y_train_single = sample_test(a=a, N=500)
#plt.plot(x_train_single,model.predict(x_train_single), label='ours')
#plt.plot(x_train_single,model_base.predict(x_train_single), label='base')
#plt.plot(x_train_single,a*np.sin(x_train_single+phi), label='truth')
#plt.xlabel('x')
#plt.xlabel('y')
#plt.legend()
#tikzplotlib.save('phi_4.tex')
#plt.show()


