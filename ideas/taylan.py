import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import os 
import cycler
import tikzplotlib

n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)


def sample_test(a=1, phi=np.pi, N=50, noise=True, sort=True, width=5):
    x = np.random.normal(0, 2, (N,1))
    if sort:
        x = np.sort(x,axis=0)
    if noise:
        y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,0.1,(N,1))
    else:
        y = np.multiply(a, np.sin(x+phi)) 
    return x, y 




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
    return np.array(sizes), train_errors, test_errors

def Error(model,X,Y,size,repeat):
    counter = 0
    inter_train_errors = []
    inter_test_errors = []
    np.random.seed(counter)
    counter+=1
    for rep in range(repeat):
        Xtrn,Ytrn,Xtst,Ytst =  splitter(X,Y,size)
        model.fit(Xtrn, Ytrn)
        inter_train_errors.append(model.compute_error(Xtrn,Ytrn))
        inter_test_errors.append(model.compute_error(Xtst,Ytst))
    return [np.mean(inter_train_errors), np.std(inter_train_errors),
            np.mean(inter_test_errors), np.std(inter_test_errors)]

def learning_curve_parallel(model,X,Y,sizes=[2,10,20,30,40],repeat=10, jobs=-1):
    results = Parallel(n_jobs=-1)\
            (delayed(Error)(model,X,Y,size,repeat) for size in sizes)
    return np.array(sizes), np.array(results)

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


class KernelRidge():
    def __init__(self, lmbda, kernel, opt=True):
        self.lmbda= lmbda
        self.jitter= 1e-8
        self.kernel = kernel
        self.opt = opt
    def fit(self, X, Y):
        if self.opt:
            self.fit_them_all(X,Y)
            self.alpha = np.linalg.inv(self.kernel(X,X) + \
                    (self.jitter+self.lmbda)*np.eye(X.shape[0])).dot(Y)
            self.X = X

    def fit_inter(self, X, Y):
        self.alpha = np.linalg.inv(self.kernel(X,X) + \
                (self.jitter+self.lmbda)*np.eye(X.shape[0])).dot(Y)
        self.X = X

    def fit_them_all(self, X, Y, grid=10):
        ls = np.linspace(0.1,1,grid)
        lmbdas = np.linspace(0.01, 10, grid)
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

    def predict(self, X):
        return (self.kernel(X, self.X)).dot(self.alpha)

    def compute_error(self, X, Y):
        return MSE(self.predict(X), Y)

class SemiParamKernelRidge():
    def __init__(self, lmbda, kernel, funcs, opt=True):
        self.lmbda= lmbda
        self.jitter= 1e-8
        self.kernel = kernel
        self.funcs = funcs 
        self.opt = opt

    def fit(self, X, Y):
        if self.opt:
            self.fit_them_all(X,Y)
            self.X = X
            self.Y = Y
            self.psi_ = self.funcs(X)
            self.optim(X,Y,self.psi_)
        else:
            self.fit_inter(X,Y)

    def fit_inter(self, X, Y):
        self.psi_ = self.funcs(X)
        self.optim(X,Y,self.psi_)
        self.Y = Y
        self.X = X

    def fit_them_all(self, X, Y, grid=10):
        ls = np.linspace(0.001,1,grid)
        lmbdas = np.linspace(0., 10, grid)
        Xtrn, Xtst, Ytrn, Ytst = train_test_split(X,Y, test_size=0.2,
                                                                random_state=2)
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
        K = self.kernel(X,X)+np.eye(X.shape[0])*self.jitter
        A = np.block([[K,psi_]])
        B = np.block([[K,np.zeros((X.shape[0],psi_.shape[1]))],
                     [np.zeros((psi_.shape[1],X.shape[0]+psi_.shape[1]))]])
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

def plot_learning_curve(estimators, X, Y, sizes=[2, 10, 20, 30, 40], repeat=20):
    fig = plt.figure()
    fig.tight_layout()
    for i,estimator in enumerate(estimators):
        ax = plt.subplot(len(estimators),1,i+1)
        train_size, scores = \
        learning_curve_parallel(estimator, X, Y, sizes=sizes, repeat=repeat)
        train_scores_mean = scores[:,0]
        train_scores_std = scores[:,1]
        test_scores_mean = scores[:,2]
        test_scores_std = scores[:,3]
        ax.set_title(r"$\bf{"+str(estimator.__class__.__name__)+"}$",fontsize=8)
        ax.fill_between(train_size,train_scores_mean - train_scores_std,
                          train_scores_mean + train_scores_std,
                          alpha=0.1)
        ax.plot(train_size, train_scores_mean, '-o',label='train_mean')
        ax.fill_between(train_size,test_scores_mean - test_scores_std,
                          test_scores_mean + test_scores_std,
                          alpha=0.1) 
        ax.plot(train_size, test_scores_mean, '--o',label='test_mean')
        #ax.set_xlim([0,1])
        ax.set_ylim([-0.1,0.5])
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
            bbox_transform=fig.transFigure, ncol=3, frameon=False)
    plt.subplots_adjust(hspace=0.3)
    fig.supxlabel('N')
    fig.supylabel('MSE')

def learning_curve_generator(l = 0.01,
                             lmbda = 0.,
                             save = False,
                             save_ext = 'tex',
                             noise = False,
                             known = True):
    
    if noise:
        noise_tag = "noisy"
    else:
        noise_tag = "noiseles"

    if known:
        known_tag = "known"
    else:
        known_tag = "unknown"   

    kernel = rbf(l=l)
    path = known_tag+noise_tag
    os.makedirs(path, exist_ok=True)
    for M in range(2,10,20):
        funcs = func_gen(num=M)
        model_base = KernelRidge(lmbda, kernel)
        model = SemiParamKernelRidge(lmbda, kernel, funcs)
        if known:
            phi = funcs.phi[0]
        else:
            phi =  np.random.normal(0,1)
        X, Y = sample_test(phi=phi, N=1000, sort=True, noise=noise)
        plot_learning_curve([model_base, model], X, Y)
        plt.suptitle(\
        "Learning Curve for {} data, {} target function and $M$={} ".format(\
        noise_tag,known_tag,M), fontsize=10)
        if save_ext == 'pdf':
            filename = 'learningcurve'+str(M)+'.pdf'
        elif save_ext == 'tex':
            filename = 'learningcurve'+str(M)+'.tex'
        filepath = os.path.join(path, filename)
        
        if save and save_ext == 'tex':
            tikzplotlib.save(filepath)
        elif save and save_ext == 'pdf':
            plt.savefig(filepath)
        else:
            plt.show()

    

        
def function_curve_generator(Ntrn=5,
                             l = 0.01,
                             lmbda = 0.,
                             save = False,
                             save_ext = 'tex',
                             noise = False,
                             known = False):

    Ms = range(1,20,1)
    repeat = 10
    Ntst = 1000
    if noise:
        noise_tag = "noisy"
    else:
        noise_tag = "noiseles"

    if known:
        known_tag = "known"
    else:
        known_tag = "unknown"   
        
    kernel = rbf(l=l)
    path = known_tag+noise_tag
    os.makedirs(path, exist_ok=True)
    train_scores_mean = []
    test_scores_mean = []
    train_scores_std = []
    test_scores_std = []

    for M in Ms:
        train_scores = []
        test_scores = []
        funcs = func_gen(num=M)
        model = SemiParamKernelRidge(lmbda, kernel, funcs)
        if known:
            phi = funcs.phi[0]
        else:
            phi =  np.random.normal(0,1)
        Xtrn, Ytrn = sample_test(phi=phi, N=Ntrn, sort=True, noise=noise)
        Xtst, Ytst = sample_test(phi=phi, N=Ntst, sort=True, noise=noise)
        for i in range(repeat):
            model.fit(Xtrn, Ytrn)
            train_scores.append(model.compute_error(Xtrn, Ytrn))
            test_scores.append(model.compute_error(Xtst, Ytst))
        train_scores_mean.append(np.mean(train_scores))
        train_scores_std.append(np.std(train_scores))
        test_scores_mean.append(np.mean(test_scores))
        test_scores_std.append(np.std(test_scores))

    train_scores_mean = np.array(train_scores_mean)
    test_scores_mean = np.array(test_scores_mean)
    train_scores_std = np.array(train_scores_std)
    test_scores_std = np.array(test_scores_std)
    train_scores_mean, train_scores_std, test_scores_mean, test_scores_std

    fig = plt.figure()
    fig.tight_layout()

    ax = plt.subplot(111)
    ax.set_title(r"$\bf{"+str(model.__class__.__name__)+"}$",fontsize=8)
    ax.fill_between(Ms, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std,
                      alpha=0.1)
    ax.plot(Ms, train_scores_mean, '-o',label='train_mean')
    ax.fill_between(Ms,test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std,
                      alpha=0.1) 
    ax.plot(Ms, test_scores_mean, '--o',label='test_mean')
    ax.set_ylim([-0.1,0.1])
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
            bbox_transform=fig.transFigure, ncol=3, frameon=False)
    plt.subplots_adjust(hspace=0.3)
    fig.supxlabel('M')
    fig.supylabel('MSE')
    plt.suptitle("M Curve for {} data, {} target function and $N$={} ".format(\
            noise_tag,known_tag,Ntrn), fontsize=10)
    if save_ext == 'pdf':
        filename = 'Mcurve'+str(Ntrn)+'.pdf'
    elif save_ext == 'tex':
        filename = 'learningcurve'+str(Ntrn)+'.tex'
    filepath = os.path.join(path, filename)
    
    if save and save_ext == 'tex':
        tikzplotlib.save(filepath)
    elif save and save_ext == 'pdf':
        plt.savefig(filepath)
    else:
        plt.show()


    
if __name__ == "__main__":
    
    noises = [False, True]
    knowns = [False, True]
    Ntrns = np.range(2,10)

    for noise in noises:
        for known  in knowns:
            learning_curve_generator(noise=noise,known=known,save=True)

    for Ntrn in Ntrns:
        for noise in noises:
            for known  in knowns:
                function_curve_generator(Ntrn=Ntrn,noise=noise,\
                                         known=known,save=True)


    
    
 


