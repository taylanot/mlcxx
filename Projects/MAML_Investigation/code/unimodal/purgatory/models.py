import numpy as np

#np.random.seed(2)
class Linear():
    def __init__(self, alpha=None):
        self.alpha = alpha

    def features(self, X):
        add = np.ones(X.shape[0]).reshape(-1,1)
        return np.hstack((X,add))

    def fit(self, X, Y):
        X = self.features(X)
        N = X.shape[0]
        if self.alpha != None:
            self. w = np.linalg.pinv(X.T.dot(X)+np.eye(X.shape[1])*self.alpha).dot(X.T.dot(Y))
        else:
            self. w = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(Y))


    def predict(self, X):
        X = self.features(X)
        return X.dot(self.w)

class Ridge(Linear):
    def __init__(self,alpha=0.01):
        super().__init__(alpha)



class SGD():
    def __init__(self, lr=0.001, n_iter=1):
        self.lr = lr
        self.n_iter = n_iter

    def features(self, X):
        add = np.ones(X.shape[0]).reshape(-1,1)
        return np.hstack((X,add))

    def init_weights(self, d):
        self.w = np.zeros(d).reshape(-1,1)


    def fit(self, X, Y):
        X = self.features(X)
        N = X.shape[0]
        self.init_weights(X.shape[1])
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def predict(self, X):
        X = self.features(X)
        return X.dot(self.w)


#        return self.model.predict(x)
        
class Bayes():
    def __init__(self, alpha=0., std_y= 5):
        jitter = 1.e-12
        self.alpha = alpha + jitter

        self.beta = 1. / (std_y**2+jitter)

    def features(self, X):
        add = np.ones(X.shape[0]).reshape(-1,1)
        return np.hstack((X,add))

    def fit(self, X,y):
        X = self.features(X)
        S0_inv = (1./self.alpha) * np.eye(X.shape[1])
        SN_inv =  S0_inv + self.beta * np.dot(X.T, X)
        self.SN = np.linalg.inv(SN_inv)
        self.mN  = self.beta * (np.dot(self.SN, (X.T).dot(y)))

    def predict(self,X):
        X = self.features(X)
        return X.dot(self.mN)

#def error(y,yp):
#    return (yp-y).T.dot((yp-y)) / (y.size)
#
#def sample_a(dim,m,c):
#    return np.random.multivariate_normal(np.ones(dim)*m, np.eye(dim)*c)
#
#def sample_x(dim, b, N):
#    return np.random.uniform(0, b, (N,dim))
#
#def sample_y(a,x,std_y):
#    return np.random.normal(0, std_y) + x.dot(a.reshape(-1,1))
#
#def sample_Z(dim, a, b, std_y, N):
#    x = sample_x(dim, b, N)
#    y = sample_y(a, x, std_y)
#    return x, y
#
#def exp_err(dim, m, c, std_y, b, Ntrn, Ntst, Nz, Na, model):
#    eea = 0
#    for j in range(Na):
#        eez = 0
#        a = sample_a(dim, m, c)
#        for i in range(Nz):
#            xtrn, ytrn = sample_Z(dim, a, b, std_y, Ntrn)
#            xtst, ytst = sample_Z(dim, a, b, std_y, Ntrn)
#
#            model.fit(xtrn,ytrn)
#            eez += (error(ytst, model.predict(xtst).reshape(-1,1)))/Nz
#        eea += (eez)/Na
#    return eea
#
#params = dict()
#params['dim'] = 1
#params['m'] = 0
#params['c'] = np.array([0.,0.5,1.,5.])
#params['std_y'] = np.linspace(0,3,20)
#params['b'] = np.linspace(1,5,5)
#params['Ntrn'] = [1,2,5,10]
#params['Ntst'] = 5000
#params['Nz'] = 100
#params['Na'] = 100 
#params['lr'] = np.linspace(1e-3,1,1)
#params['alpha'] = np.linspace(0,5,1)
#
#
#model_R = Ridge()
#model_L = Linear()
#model_S = SGD()
#model_B = Bayes()
#
#print(exp_err(params['dim'], params['m'], 0, 3, 1, 5, params['Ntst'], params['Nz'], params['Na'], model_S))
#print(exp_err(params['dim'], params['m'], 0, 3, 1, 5, params['Ntst'], params['Nz'], params['Na'], model_R))
#print(exp_err(params['dim'], params['m'], 0, 3, 1, 5, params['Ntst'], params['Nz'], params['Na'], model_L))
