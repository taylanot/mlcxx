import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
import tikzplotlib
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon, friedmanchisquare, ttest_ind, f_oneway, \
                        brunnermunzel, mannwhitneyu,cramervonmises_2samp

n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

np.random.seed(24)

def MSE(x,x_):
    assert x.shape == x_.shape
    return ((x-x_)**2).mean()
    
def sample(a=100, phi=np.pi, N=50, noise=True, sort=True, width=5,test=False):
    if test:
        x = np.random.normal(0, 5, (N,1))
    else:
        x = np.random.normal(0, 5, (N,1))
    if sort:
        x = np.sort(x,axis=0)
        if noise:
            y = np.multiply(a, np.sin(x+phi)) + np.random.normal(0,.1,(N,1))
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

    def fit_them_all(self, X, Y, grid=10):
        ls = np.linspace(0.01,1,grid)
        lmbdas = np.linspace(0.01, 1, grid)
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

    def fit_them_all(self, X, Y, grid=20):
        ls = np.linspace(0.01,1,grid)
        lmbdas = np.linspace(0.0001, 1, grid)
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


    def optim(self, X, Y, psi_):
        K = self.kernel(X,X)
        K += np.eye(K.shape[0])*1e-6
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
        self.a = np.random.normal(1,0.1,num)
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
        #return np.hstack(([np.multiply(self.a, np.tan(x+self.phi))]))

### Block-2 ####

#a =  1. #*np.random.normal(1,1)
#
#lmbda = 0.
#l = 0.01 
#N = 10
#Ntst = 1000
#repeat = 100
#M = 10
#
#kernel = rbf(l=l)
#funcs = func_gen(num=M)
#
#phi =  np.random.normal(0,1)
#
##noise = True
#noise = False
#
#x_train_single, y_train_single = sample_test(a=a, phi=phi, N=N,sort=True,noise=noise)
#
#testsets= []
#for i in range(repeat):
#    testsets.append(sample_test(a=a, phi=phi, N=Ntst,sort=True,noise=noise))
#
#model = SemiParamKernelRidge(lmbda, kernel, funcs)
#model_null = SemiParamKernelRidge(lmbda, kernel, funcs, null=True)
#model.fit(x_train_single,y_train_single)
#model_null.fit(x_train_single,y_train_single)
#
#err_null = []
#err = []
#
#for dataset in testsets:
#    err.append(model.error(dataset[0], dataset[1]))
#    err_null.append(model_null.error(dataset[0], dataset[1]))
#
#H0 = np.array(err_null)
#H1 = np.array(err)
#
#print(H0.mean())
#print(H1.mean())
#
#from scipy.stats import wilcoxon, friedmanchisquare, ttest_ind
##print(wilcoxon(H0-H1))
#print(ttest_ind(H0, H1))

a =  1. #*np.random.normal(1,1)

lmbda = 0.1
l = 0.1 
N = 5
Ntst = 10000
repeat = 100
M = 10

kernel = rbf(l=l)
funcs = func_gen(num=M)

phi =  np.random.normal(0,1)

#noise = True
noise = False
print("noise:{}".format(noise))

err0 = []
err1 = []
err2 = []


dataset = sample(a=a, phi=phi, N=Ntst,sort=True,noise=noise,test=True)
for case in range(repeat):
    x, y = sample(a=a, phi=phi, N=N,sort=True,noise=noise)
    model0 = SemiParamKernelRidge(lmbda, kernel, funcs, null=True)
    model1 = SemiParamKernelRidge(lmbda, kernel, funcs)
    model2 = KernelRidge(lmbda, kernel)

    #model0.fit_inter(x,y)
    #model1.fit_inter(x,y)
    #model2.fit_inter(x,y)

    model0.fit(x,y)
    model1.fit(x,y)
    model2.fit(x,y)

    err0.append(model0.error(dataset[0], dataset[1]))
    err1.append(model1.error(dataset[0], dataset[1]))
    err2.append(model2.error(dataset[0], dataset[1]))

H0 = np.array(err0)
H1 = np.array(err1)
H2 = np.array(err2)

print("std0:{}".format(np.std(H0)))
print("std1:{}".format(np.std(H1)))
print("std2:{}".format(np.std(H2)))

bins = 10
plt.hist(H0,bins)
plt.hist(H1,bins)
plt.hist(H2,bins)
#plt.show()
#print("W/0-1:",wilcoxon(H0,H1))
#print("W/0-2:",wilcoxon(H0,H2))
#print("W/1-2:",wilcoxon(H1,H2))
#print("W/0-1:",brunnermunzel(H0,H1))
#print("W/0-2:",brunnermunzel(H0,H2))
#print("W/1-2:",brunnermunzel(H1,H2))

#print("W/0-1:",mannwhitneyu(H0,H1))
#print("W/0-2:",mannwhitneyu(H0,H2))
#print("W/1-2:",mannwhitneyu(H1,H2))

print("W/0-1:",cramervonmises_2samp(H0,H1))
print("W/0-2:",cramervonmises_2samp(H0,H2))
print("W/1-2:",cramervonmises_2samp(H1,H2))

#print("f/0-1:",f_oneway(H0,H1))
#print("f/0-2:",f_oneway(H0,H2))
#print("f/1-2:",f_oneway(H1,H2))
#print("t/0-1:",ttest_ind(H0,H1))
#print("t/0-2:",ttest_ind(H0,H2))


