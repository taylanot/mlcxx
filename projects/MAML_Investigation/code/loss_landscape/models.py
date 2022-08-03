import numpy as np
import matplotlib.pyplot as plt
import copy as cp

def MSE(x,x_):
    assert x.shape == x_.shape
    return ((x-x_)**2).mean()
 
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


        return self.model.predict(x)


if __name__=='__main__':

    np.random.seed(24)
    dim = 1
    Ntrn = 1
    Ntst = 10000

    x_train = np.random.normal(0,1,(Ntrn,1))
    x_test = np.random.normal(0,1,(Ntst,1))
    y_train = x_train #+ np.random.normal(0,1,(Ntrn,1))
    y_test = x_test #+ np.random.normal(0,1,(Ntst,1))

    model = Linear(alpha=None)


    model.fit(x_train, y_train)
    w1_trn, w2_trn  = cp.deepcopy(model.w[0]), cp.deepcopy(model.w[1])

    model.fit(x_test, y_test)
    w1_tst, w2_tst  = cp.deepcopy(model.w[0]), cp.deepcopy(model.w[1])

    w1= 1
    w2= 0 
    window = 5 
    resolution = 200

    w1s, w2s = np.meshgrid(np.linspace(w1-window, window+w1, resolution),np.linspace(w2-window, w2+window, resolution))

    train_loss = np.zeros(w1s.shape)
    test_loss = np.zeros(w1s.shape)

    for i in range(resolution):
        for j in range(resolution):
            model.w[0] = w1s[j,i]
            model.w[1] = w2s[j,i]
            train_loss[i,j] = MSE(y_train, model.predict(x_train))
            test_loss[i,j] = MSE(y_test, model.predict(x_test))
            
    fig,ax=plt.subplots(2, figsize=(10,10))
    cp0 = ax[0].contourf(w1s, w2s, train_loss)
    ax[0].scatter(w1_trn,w2_trn,c='r', marker="x")
    cp1 = ax[1].contourf(w1s, w2s, test_loss)
    ax[1].scatter(w1_tst,w2_tst,c='r', marker="x")

    plt.colorbar(cp0, ax=ax[0]) # Add a colorbar to a plot
    plt.colorbar(cp1, ax=ax[1]) # Add a colorbar to a plot
    for i in range(2):
        ax[i].set_title('Loss Landscape')
        ax[i].set_xlabel('$w_1$')
        ax[i].set_ylabel('$w_2$')
    plt.show()
    #model = Linear(alpha=None)

    #model.fit(x,y)
    #print(model.w)

