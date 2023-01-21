import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import cycler
#import tikzplotlib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import os 
#n = 8
#color = plt.cm.Dark2(np.linspace(0, 1,n))
#mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

np.random.seed(24)

def splitter(X,Y, size=10):
    i = np.random.choice(X.shape[0],size,replace=False)
    return X[i,:], Y[i], np.delete(X,i, axis=0), np.delete(Y,i, axis=0)

def learning_curve(model,X,Y,sizes=[2,10,20,30,40],repeat=10):
    train_errors = []
    test_errors = []
    for size in sizes:
        inter_train_errors = []
        inter_test_errors = []
        for rep in range(repeat):
            Xtrn,Ytrn,Xtst,Ytst =  splitter(X,Y,size)
            model.fit(Xtrn, Ytrn)
            inter_train_errors.append(model.compute_error(Xtrn,Ytrn))
            inter_test_errors.append(model.compute_error(Xtst,Ytst))
        train_errors.append(inter_train_errors)
        test_errors.append(inter_test_errors)
    print(test_errors)
    return np.array(sizes), train_errors, test_errors

def MSE(x,x_):
    assert x.shape == x_.shape
    return ((x-x_)**2).mean()
    
def sample_train(Na=1,N=50):
    a = np.random.normal(1, 1, (1,Na))
    phi = np.random.normal(0, 1, (1,Na))
    x = np.random.normal(0, 1, (N,Na))
    y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,0.1,(N,Na))
    return x, y , a

def sample_test(a=100, phi=np.pi, N=50, noise=True, sort=True, width=5):
    x = np.random.normal(0, 2, (N,1))
    if sort:
        x = np.sort(x,axis=0)
    if noise:
        y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,0.1,(N,1))
    else:
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


class KernelRidge(BaseEstimator):
    def __init__(self, lmbda, kernel):
        self.lmbda= lmbda
        self.kernel = kernel
        self.jitter = 1e-8

    def fit(self, X, Y):
        self.fit_them_all(X,Y)
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
                (self.jitter+self.lmbda)*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_inter(self, X, Y):
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
                (self.jitter+self.lmbda)*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_them_all(self, X, Y, grid=20):
        ls = np.linspace(0.001,1,grid)
        lmbdas = np.linspace(0., 10, grid)
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

    def compute_error(self, X, Y):
        return MSE(self.predict(X), Y)


class SemiParamKernelRidge(BaseEstimator):
    def __init__(self, lmbda, kernel, funcs):
        self.lmbda= lmbda
        self.kernel = kernel
        self.funcs = funcs 
        self.jitter = 0.#1.e-8

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

    def fit_them_all(self, X, Y, grid=20):
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
        K = self.kernel(X,X) + self.jitter*np.eye(X.shape[0])
        A = np.block([[K,psi_]])
        B = np.block([[K,np.zeros((X.shape[0],psi_.shape[1]))],[np.zeros((psi_.shape[1],X.shape[0]+psi_.shape[1]))]])
        
        self.w = np.linalg.pinv(A.T.dot(A)+self.lmbda+B.T).dot(A.T.dot(Y))


    def predict(self, X):
        psi_ = self.funcs(X)
        A = np.block([[self.kernel(X, self.X),psi_]])
        return A.dot(self.w)

    def compute_error(self, X, Y):
        return MSE(self.predict(X), Y)


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
        return np.hstack(([np.multiply(self.a, np.sin(x+self.phi))]))

def plot_learning_curve(estimators):
    ax = [plt.subplot(211), plt.subplot(212)]
    for i,estimator in enumerate(estimators):
        train_size, train_scores, test_scores = \
        learning_curve(estimator, x_train_single, y_train_single,\
        sizes=[2, 10, 20, 30, 40], repeat=20)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        ax[i].fill_between(train_size,train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std,
                             alpha=0.1)
        ax[i].plot(train_size, train_scores_mean, label='train_mean')
        ax[i].fill_between(train_size,test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std,
                             alpha=0.1) 
        ax[i].plot(train_size, test_scores_mean, label='test_mean')

if __name__ == "__main__":
   # create_M_curves()
    a =  1.

    l = 0.01
    lmbda = 0.

    noise = False
    if noise:
        noise_tag = "noisy"
    else:
        noise_tag = "noiseles"
        
    kernel = rbf(l=l)
    #path = "known_"+noise_tag+"/l"+str(l)+"-lmbda"+str(lmbda)
    path = "unknown_"+noise_tag
    os.makedirs(path, exist_ok=True)
    for M in range(2,100,20):
        funcs = func_gen(num=M)
        model_base = KernelRidge(lmbda, kernel)
        model = SemiParamKernelRidge(lmbda, kernel, funcs)
        #phi =  np.random.normal(0,1)
        phi = funcs.phi[0]
        x_train_single, y_train_single = sample_test(a=a, phi=phi, N=1000, sort=True, noise=noise)
        plt.figure()
        plot_learning_curve([model_base, model])
        filename = 'learning'+str(M)+'.pdf'
        filepath = os.path.join(path, filename)
        plt.legend()
        plt.xlabel('N')
        plt.ylabel('MSE')
        #plt.savefig(filepath)
        plt.show()

#def create_M_curves():
#    a =  1.#np.random.normal(1,1)
#    Ms = range(1,20,1)
#    l = 0.01
#    lmbda = 0.
#    Ntrn = 5
#    Ntst = 1000
#    repeat = 20
#    noise = True
#    if noise:
#        noise_tag = "noisy"
#    else:
#        noise_tag = "noiseles"
#        
#    kernel = rbf(l=l)
#    #path = "known_"+noise_tag+"/l"+str(l)+"-lmbda"+str(lmbda)
#    path = "known_"+noise_tag
#    os.makedirs(path, exist_ok=True)
#
#    
#    train_scores_mean = []
#    test_scores_mean = []
#    train_scores_std = []
#    test_scores_std = []
#
#    for M in Ms:
#        train_scores = []
#        test_scores = []
#
#        funcs = func_gen(num=M)
#        phi = funcs.phi[0]
#        #phi =  np.random.normal(0,1)
#        for i in range(repeat):
#            x_train_single, y_train_single = sample_test(a=a, phi=phi, N=Ntrn, sort=True, noise=noise)
#            x_test_single, y_test_single = sample_test(a=a, phi=phi, N=Ntst, sort=True, noise=noise)
#            model = SemiParamKernelRidge(lmbda, kernel, funcs)
#            model.fit(x_train_single, y_train_single)
#            train_scores.append(MSE(model.predict(x_train_single), y_train_single))
#            test_scores.append(MSE(model.predict(x_test_single), y_test_single))
#        train_scores_mean.append(np.mean(train_scores))
#        train_scores_std.append(np.std(train_scores))
#        test_scores_mean.append(np.mean(test_scores))
#        test_scores_std.append(np.std(test_scores))
#
#    train_scores_mean = np.array(train_scores_mean)
#    test_scores_mean = np.array(test_scores_mean)
#    train_scores_std = np.array(train_scores_std)
#    test_scores_std = np.array(test_scores_std)
#    ax = plt.subplot(111)
#    ax.fill_between(Ms,train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std,
#                     alpha=0.1)
#    ax.plot(Ms, train_scores_mean, label='train_mean')
#    ax.fill_between(Ms,test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std,
#                      alpha=0.1) 
#    ax.plot(Ms, test_scores_mean, label='test_mean')
#    filename = 'learning_M.pdf'
#    filepath = os.path.join(path, filename)
#    plt.legend()
#    plt.title("N={}".format(Ntrn))
#    plt.xlabel('M')
#    plt.ylabel('MSE')
#    plt.savefig(filepath)
#    plt.show()
#
#def create_learning_curves():
#    a =  1.
#
#    l = 0.01
#    lmbda = 0.
#
#    noise = True 
#    if noise:
#        noise_tag = "noisy"
#    else:
#        noise_tag = "noiseles"
#        
#    kernel = rbf(l=l)
#    #path = "known_"+noise_tag+"/l"+str(l)+"-lmbda"+str(lmbda)
#    path = "unknown_"+noise_tag
#    os.makedirs(path, exist_ok=True)
#    for M in range(2,100,20):
#        funcs = func_gen(num=M)
#        model_base = KernelRidge(lmbda, kernel)
#        model = SemiParamKernelRidge(lmbda, kernel, funcs)
#        phi =  np.random.normal(0,1)
#        #phi = funcs.phi[0]
#        x_train_single, y_train_single = sample_test(a=a, phi=phi, N=1000, sort=True, noise=noise)
#        plt.figure()
#        plot_learning_curve([model_base, model])
#        filename = 'learning'+str(M)+'.pdf'
#        filepath = os.path.join(path, filename)
#        plt.legend()
#        plt.title("M={}".format(M))
#        plt.xlabel('N')
#        plt.ylabel('MSE')
#        #plt.savefig(filepath)
#        plt.show()

