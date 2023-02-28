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
    return x, y , a

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

class SemiParamKernelRidge():
    def __init__(self, lmbda, kernel, funcs):
        self.lmbda= lmbda
        self.kernel = kernel
        self.funcs = funcs 

    def fit(self, X, Y):
        self.fit_them_all(X,Y)
        self.X = X
        self.Y = Y
        self.psi_ = np.hstack(([func(X) for func in self.funcs]))
        self.optim(X,Y,self.psi_)

    def fit_inter(self, X, Y):
        self.psi_ = np.hstack(([func(X) for func in self.funcs]))
        self.optim(X,Y,self.psi_)
        self.Y = Y
        self.X = X

    def fit_them_all(self, X, Y, grid=10):
        ls = np.linspace(self.kernel.l-0.9, self.kernel.l+0.9,grid)
        lmbdas = np.linspace(0, 100, grid)
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

    def optim(self, X, Y, psi_, itr=1000):
        self.alpha = np.zeros(X.shape[0]).reshape(-1,1)
        self.beta = np.zeros(len(self.funcs)).reshape(-1,1)
        K = self.kernel(X,X)
        for i in range(itr):
            self.alpha = np.linalg.pinv( K + self.lmbda*np.eye(X.shape[0])).\
                    dot(Y-psi_.dot(self.beta))
            self.beta = np.linalg.pinv(psi_).dot(Y-K.dot(self.alpha))
            print("alpha:\n",self.alpha)
            print("beta:\n",self.beta)

    def predict(self, X):
        psi_ = np.hstack(([func(X) for func in self.funcs]))
        return (self.kernel(X, self.X)).dot(self.alpha) + psi_.dot(self.beta)

class SemiParamKernelRidge2():
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
        lmbdas = np.linspace(0, 0.1, grid)
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
        print(self.kernel.l)
        print(self.lmbda)


    def optim(self, X, Y, psi_, itr=100):
        self.alpha = np.zeros(X.shape[0]).reshape(-1,1)
        self.beta = (np.ones(len(self.funcs)).reshape(-1,1))
        #self.beta = np.random.rand(len(self.funcs),1)
        K = self.kernel(X,X)
        for i in range(itr):
            self.alpha = np.linalg.pinv( K + self.lmbda*np.eye(X.shape[0])).\
                    dot(Y)
            self.beta = np.linalg.pinv(psi_).dot(Y-K.dot(self.alpha))
            #print("alpha:\n",self.alpha)
            #print("beta:\n",self.beta)


    def predict(self, X):
        psi_ = self.funcs(X)
        return (self.kernel(X, self.X)).dot(self.alpha) + psi_.dot(self.beta)

class SemiParamKernelRidge3():
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
        #self.kernel.l = ls[index[0][0]]
        #self.lmbda = lmbdas[index[1][0]]
        self.kernel.l = 0.1
        self.lmbda = 0.

        print(self.kernel.l)
        print(self.lmbda)


    def optim(self, X, Y, psi_):
        K = self.kernel(X,X)
        psi_inv  = np.linalg.inv((psi_.T).dot(psi_)+(1e-8)*np.eye(psi_.shape[1]))

        #self.alpha = np.dot(np.linalg.inv(K - psi_.dot(psi_inv).dot(psi_.T).dot(K) + (self.lmbda+1e-6)*np.eye(X.shape[0])) \
        #         ,(Y - np.dot(psi_.dot(psi_inv), psi_.T.dot(Y))))

        #self.beta = np.dot(psi_inv, (psi_.T.dot(Y) - psi_.T.dot(K.dot(Y))))

        KK = K.dot(K)
        self.alpha = np.dot(np.linalg.inv(KK + self.lmbda*K - K.dot(psi_.dot(psi_inv.dot(psi_.T.dot(K))))), \
                            (K.dot(psi_.dot(psi_inv.dot(psi_.T))) - K).dot(Y))

        self.beta = np.dot(psi_inv, psi_.T.dot(Y)- psi_.T.dot(K.dot(self.alpha)))

    def predict(self, X):
        psi_ = self.funcs(X)
        return (self.kernel(X, self.X)).dot(self.alpha) + psi_.dot(self.beta)
        #return (self.kernel(X, self.X)).dot(self.alpha) 
        #return psi_.dot(self.beta)

    def predict2(self, X):
        return (self.kernel(X, self.X)).dot(self.alpha) 

class SemiParamKernelRidge4():
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


    def optim(self, X, Y, psi_):
        K = self.kernel(X,X)
        A = np.block([[K,psi_]])
        B = np.block([[K,np.zeros((X.shape[0],psi_.shape[1]))],[np.zeros((psi_.shape[1],X.shape[0]+psi_.shape[1]))]])
        
        #psi_inv  = np.linalg.inv((psi_.T).dot(psi_)+(1e-8)*np.eye(psi_.shape[1]))

        #self.alpha = np.dot(np.linalg.inv(K - psi_.dot(psi_inv).dot(psi_.T).dot(K) + (self.lmbda+1e-6)*np.eye(X.shape[0])) \
        #         ,(Y - np.dot(psi_.dot(psi_inv), psi_.T.dot(Y))))

        #self.beta = np.dot(psi_inv, (psi_.T.dot(Y) - psi_.T.dot(K.dot(Y))))

        #KK = K.dot(K)
        #self.alpha = np.dot(np.linalg.inv(KK + self.lmbda*K - K.dot(psi_.dot(psi_inv.dot(psi_.T.dot(K))))), \
        #                    (K.dot(psi_.dot(psi_inv.dot(psi_.T))) - K).dot(Y))

        #self.beta = np.dot(psi_inv, psi_.T.dot(Y)- psi_.T.dot(K.dot(self.alpha)))
        self.w = np.linalg.inv(A.T.dot(A)+(1e-6+self.lmbda)*B.T).dot(A.T.dot(Y))


    def predict(self, X):
        psi_ = self.funcs(X)
        
        A = np.block([[self.kernel(X, self.X),psi_]])
        return A.dot(self.w)
        #return (self.kernel(X, self.X)).dot(self.alpha) + psi_.dot(self.beta)
        #return (self.kernel(X, self.X)).dot(self.alpha) 
        #return psi_.dot(self.beta)

class func_gen():
    def __init__(self, num=10):
        self.num = num
        self.a = 1.#np.random.normal(1,0.1,num)
        self.phi = np.random.normal(0,1,num)
        counter = 0 

    def __len__(self):
        return self.num

    def __iter__(self):
        return np.sin

    def __call__(self, x):
        return np.hstack(([np.multiply(self.a, np.sin(x+self.phi))]))



#x_train, y_train, _ = sample_train()
#
#x_test, y_test = sample_test(a=1, N=1000)
#
#kernel = rbf()
#model = KernelRidge(1., kernel)
#model.fit(x_train,y_train)
#
#print(MSE(model.predict(x_test),y_test))

### Block-1 ####
#a = 4.# np.random.normal(1,1)
#phi = 0
#x_train_single, y_train_single = sample_test(a=a,phi=phi, N=4)
#kernel = rbf()
#
#funcs = [np.sin]#, np.tan, np.tanh]
#model = SemiParamKernelRidge(100., kernel, funcs)
#model_base = KernelRidge(1, kernel)
#
#model.fit(x_train_single,y_train_single)
#model_base.fit(x_train_single,y_train_single)
#
#plt.scatter(x_train_single, y_train_single)
#x_train_single, y_train_single = sample_test(a=a,phi=phi, N=500)
#plt.plot(x_train_single,model.predict(x_train_single), label='ours')
#plt.plot(x_train_single,model_base.predict(x_train_single), label='base')
#plt.plot(x_train_single,a*np.sin(x_train_single+phi), label='truth')
#plt.xlabel('x')
#plt.xlabel('y')
#plt.legend()
#plt.show()

### Block-2 ####
a =  1. #*np.random.normal(1,1)

kernel = rbf()
funcs = func_gen(num=2)

phi =  np.random.normal(0,1)
#phi =  funcs.phi[0]

noise = False

x_train_single, y_train_single = sample_test(a=a, phi=phi, N=10,sort=True,noise=noise)


model = SemiParamKernelRidge3(1., kernel, funcs)
model2 = SemiParamKernelRidge4(1., kernel, funcs)
model_base = KernelRidge(1, kernel)

model.fit(x_train_single,y_train_single)
model2.fit(x_train_single,y_train_single)
model_base.fit(x_train_single,y_train_single)
#print(model.alpha)
#print(model.beta)
#print(model2.w)

plt.scatter(x_train_single, y_train_single)
x_train_single, y_train_single = sample_test(a=a, phi=phi, N=10000, width=20)
plt.plot(x_train_single,model2.predict(x_train_single), label='meta')
plt.plot(x_train_single,model_base.predict(x_train_single), label='standard')
plt.plot(x_train_single,a*np.sin(x_train_single+phi), ':', label='sine')
plt.xlabel('x')
plt.xlabel('y')
plt.legend()
#tikzplotlib.save('phi_4.tex')
plt.show()



#for i in range(x_train.shape[1]):
#    plt.scatter(x_test[:,i], y_test[:,i])
#plt.plot(x_test,model.predict(x_test))
#plt.show()
