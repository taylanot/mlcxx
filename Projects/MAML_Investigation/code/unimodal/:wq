import numpy as np

def features(X):
    add = np.ones(X.shape[0]).reshape(-1,1)
    return np.ascontiguousarray(np.hstack((X,add)))

def set_bias(X):
    add = np.zeros(1).reshape(-1,1)
    return np.ascontiguousarray(np.vstack((X,add)))


class SGD():
    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter

    def init_weights(self, d):
        self.w = np.random.normal(0,1,(d+1,1))

    def set_weights(self,w):
        self.w = set_bias(w)

    def fit(self, X, Y, set):
        X = features(X)
        N = X.shape[0]
        self.set_weights(set)
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = 2 * X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def fit_random(self, X, Y):
        X = features(X)
        N = X.shape[0]
        self.init_weights(X.shape[1])
        for i in range(self.n_iter):
            error = X.dot(self.w) - Y
            grad = X.T.dot(error) / N
            self.w = self.w - self.lr * grad

    def predict(self, X):
        X = features(X)
        return X.dot(self.w)


def CustomGradientDescentRegressor(m, X,Y,learning_rate=1,n_itr=1):
    w_cur=m*np.ones(shape=(1,X.shape[1])).reshape(-1,1)
    b_cur=0
    cur_itr=1
    print(w_cur,b_cur)
    w_old=w_cur
    b_old=b_cur
    w_temp=np.zeros(shape=(1,X.shape[1]))
    b_temp=0
    for i in range(X.shape[0]):
        w_temp+=X[i]*(Y[i]-(np.dot(w_old,X[i])+b_old))*(-2/X.shape[0])
        b_temp+=(Y[i]-(np.dot(w_old,X[i])+b_old))*(-2/X.shape[0])
    w_cur=w_old-learning_rate*w_temp
    b_cur=b_old-learning_rate*b_temp
    print(w_cur,b_cur)

class LinearRegression():
    def __init__(self,dim, m, b=0, lr=1.): 
        self.lr=lr
        self.w=np.array([[m]])
        self.b=np.array([b])

    def cost(self,x,y):     
        pred = x@self.w+self.b  # predicted y-values
        e=y-pred             # error term
        return np.mean(e**2)  # mean squared error

    def fit(self, x,y):
        pred = x@self.w+self.b
        e=y-pred
        dJ_dw=(np.mean(e*(-2*x), axis=0)) # partial derivate of J with respect to w
        dJ_db=(np.mean(e*(-2),axis=0)) # partial derivate of J with respect to b
        self.w = (self.w.T-self.lr*dJ_dw).T  # update w
        self.b = self.b - self.lr*dJ_db    # update b

    def predict(self, x):
        return (x @ self.w.T + self.b)  # return predicted values

    def params(self):
        return (self.w,self.b)   # return parameters
dim = 1
N = 100
m = 0
c = 100
b = 100
a = 2
seed = 10

np.random.seed(seed)
def foo(x):
  return np.dot(x,a*np.ones(x.shape[1]).reshape(-1,1)) + np.random.normal(0,1,(x.shape[0],1))

x_train = np.random.multivariate_normal(np.ones(dim)*m, np.eye(dim)*c,N)
y_train = foo(x_train)

model = SGD(1,1)
model2 = LinearRegression(dim, a)
model.fit(x_train.reshape(-1,1),y_train.reshape(-1,1),a*np.ones(dim).reshape(-1,1))
print(model.w)
print(model2.params())
model2.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))
print(model2.params())

CustomGradientDescentRegressor(a, x_train, y_train)
