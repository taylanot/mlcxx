import numpy as np
import GPy
import matplotlib.pyplot as plt
class sample_a():
    def __init__(self, dim, mean, cov):
        self.dim = dim
        self.dist = np.random.multivariate_normal
    def sample(self, N):
        return self.dist(np.ones(dim)*mean, np.eye(dim)*cov, N)

class sample_data():
    def __init__(self, dim, cov):
        self.dist = np.random.multivariate_normal
        self.dim = dim
        self.cov = cov
    def sample(self,a, N):
        x = self.dist(np.ones(dim)*0, np.eye(dim)*self.cov, N)
        #y = x @ a.T + np.random.normal(0,1,(N,1))
        y = a*np.sin(x)  + np.random.normal(0,1,(N,1))
        return x, y



class Bayes_textbook():
    def __init__(self, m=0, alpha=1, std_y=1):
        jitter = 1.e-12
        self.alpha = alpha 
        self.std_y= std_y
        if self.alpha == 0.:
            self.alpha += jitter
        if self.std_y == 0.:
            self.beta = 1. / (self.std_y**2+jitter)
        else:
            self.beta = 1. / (self.std_y**2+jitter)
        self.m = m
    def fit(self, X,y):
        X = np.sin(X)
        S0_inv = (1./self.alpha) * np.eye(X.shape[1])
        SN_inv =  S0_inv + self.beta * np.dot(X.T, X)
        self.SN = np.linalg.inv(SN_inv)
        self.mN = (np.dot(self.SN,(S0_inv.dot(self.m*np.ones(X.shape[1]).reshape(-1,1))+self.beta*(X.T).dot(y))))
    def predict(self,X):
        return np.sin(X).dot(self.mN)


dim = 1
mean = 5
cov = 100
xs = []
ys = []

xpred = np.linspace(-4,4,200).reshape(-1,1)
a_dist = sample_a(dim, mean, cov)
data_dist = sample_data(dim, cov=1)
models = []
kern = GPy.kern.RBF(dim)
for i in range(2):
    x,y = data_dist.sample(a_dist.sample(1),50)
    #model = GPy.models.GPRegression(x,y,kern)
    #model.optimize()
    #mx, _ = model.predict(xpred)
    #models.append(mx)
    xs.append(x); ys.append(y)
X = np.concatenate(xs)
Y = np.concatenate(ys)
#model = GPy.models.GPRegression(X,Y,kern)
#model.optimize()
#model.plot()
model = Bayes_textbook()
model.fit(X,Y)
print(model.mN)
plt.plot(xpred,model.predict(xpred), color='r')
plt.scatter(X,Y)
#plt.plot(xpred,sum(models)/len(models), color='r')
#mx, _ = model.predictive_gradients(x)
#print(np.mean(mx))
plt.show()


