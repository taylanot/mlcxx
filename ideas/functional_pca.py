import numpy as np
import matplotlib.pyplot as plt 
import scipy.integrate


def foo(x):
    a = np.random.normal(0,1,(num_func,1))
    xs = np.tile(x, (num_func,1))
    ys = np.array([np.sin(x+ai) for ai in a])
    #ys = np.array([ai*x**3 for ai in a])
    return x, ys #a * xs**2


def integration_weights_(x):
    #print(len(x))
    #print(scipy.integrate.simps(np.eye(len(x)),x))
    #return scipy.integrate.simps(np.eye(len(x)),x)
    return 0.5 * np.concatenate((np.array([x[1] - x[0]]),
                              x[2:] - x[:(len(x) - 2)],
                              np.array([x[len(x) - 1] - x[len(x) - 2]]))
                              ,axis=None)

def eig(X,Y, mean_sub):
    n_obs = X.size
    argvals = X.copy()
    data = Y.copy()
    weight = integration_weights_(argvals)
    #print(weight)
    # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
    weight_sqrt = np.diag(np.sqrt(weight))
    weight_invsqrt = np.diag(1 / np.sqrt(weight))
    mean = Y.mean(axis=0)
    if mean_sub:
        data -= mean
    covariance = data.T.dot(data) / (n_obs-1)
    covariance += covariance.T
    covariance /= 2
    var = np.dot(np.dot(weight_sqrt, covariance), weight_sqrt)

    eigenvalues, eigenvectors = np.linalg.eigh(var)
    eigenvalues[eigenvalues < 0] = 0
    eigenvalues = eigenvalues[::-1]
    #print(eigenvalues)
    # Slice eigenvalues and compute eigenfunctions = W^{-1/2}U
    exp_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    #print(exp_variance)
    npc = np.sum(exp_variance < 0.99) + 1

    eigenvalues = eigenvalues[:npc]
    eigenfunctions = np.transpose(np.dot(weight_invsqrt,
                                         np.fliplr(eigenvectors)[:, :npc]))
    # (NxN @ Nxnpc)^T -> npcxN

    return eigenvalues, eigenfunctions, npc
        
if __name__ == "__main__":
    num_func = 2000
    np.random.seed(24)

    dense = 600
    x = np.random.normal(0,1,dense)
    x = np.sort(x); X,Y = foo(x);
    #x = np.arange(1,6); X =x; Y=np.array([[1.,2.,3.,4.,5.],[2.,4.,6.,8.,10.]]); 
    #x = np.sort(np.random.uniform(-5,5, dense)); X,Y = foo(x);

    #print(X,Y)

    eigenvalues, eigenfunctions, npc = eig(X,Y, True)
    [plt.plot(X, eigenfunctions[i]) for i in range(npc)]
    eigenvalues, eigenfunctions, npc = eig(X,Y, False)
    [plt.plot(X, eigenfunctions[i]) for i in range(npc)]

    plt.show()


#import pandas as pd
#
#from FDApy.preprocessing.dim_reduction.fpca import UFPCA
#from FDApy.visualization.plot import plot
#from FDApy.representation.functional_data import DenseFunctionalData
#from FDApy.misc.loader import read_csv
#from FDApy.misc.utils import integration_weights_
#
##temperature = read_csv('/home/taylanot/Desktop/sandbox/FDApy/examples/data/canadian_temperature_daily.csv', index_col=0)
#
#temperature = DenseFunctionalData({'input_dim_0':X}, Y)
#
#
## Perform a univariate FPCA on dailyTemp.
#fpca = UFPCA(n_components=0.99)
#fpca.fit(temperature)
#
## Plot the results of the FPCA (eigenfunctions)
##[plt.plot(X, Y[i],'--') for i in range(num_func)]
#print(fpca.eigenvalues)
#_ = plot(fpca.eigenfunctions)
#plt.show()
