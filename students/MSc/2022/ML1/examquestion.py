import numpy as np
np.random.seed(24)

def Bayes(alpha, beta, X, Y):
    cov = X.T.dot(X)
    Sn_inv = alpha*np.eye(cov.shape[0]) + beta * cov
    Sn = np.linalg.inv(Sn_inv)
    mn = beta * Sn.dot(X.T.dot(Y))
    return mn, Sn

def posterior(beta, mn, Sn, x):
    return x.dot(mn), 1/beta + x.dot(Sn).dot(x.T)

def posterior_mean(mn, x):
    return x.dot(mn)


X = np.ones((2,2)); X[0,0] = -1
Y = np.ones((2,1)); Y[0,0] = -1
#X = np.linspace(-1,1,50).reshape(-1,1); X = np.hstack((X,np.ones((X.shape[0],1))))
#Y = np.linspace(-1,1,50).reshape(-1,1)
Xtst = np.ones((3,2)); Xtst[0,0] = -2; Xtst[1,0] = 0; Xtst[2,0] = 2
Ytst = np.ones((3,1))*2; Ytst[0,0] = -2; Ytst[1,0] = 0;

alpha = 1e-6
noise = 1.
beta = 1/noise

print(Bayes(alpha, beta, X, Y))

mn, Sn = Bayes(alpha, beta, X, Y)


#print(np.mean((posterior_mean(mn, Xtst) - Ytst)**2))
print((posterior(beta, mn,Sn, Xtst[0].reshape(1,-1))))

#samples = 100000
#dist = np.random.multivariate_normal(mn.flatten(), Sn, samples)
##print(np.mean(dist, axis=0))
#error = []
#for i in range(samples):
#   error.append(np.mean((posterior_mean(dist[i].reshape(-1,1), Xtst) - Ytst)**2,axis=0))
##print(sum(error)/len(error))
##print(dist[0])
#print(np.var(error))



#print(X.T.dot(Sn.dot(X)))




