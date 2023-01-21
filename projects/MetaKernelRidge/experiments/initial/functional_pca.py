import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os 
import cycler
import tikzplotlib
from csaps import csaps

n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)



def foo(x, num_func=100):
    a = np.random.normal(0,1,(num_func,1))
    xs = np.tile(x, (num_func,1))
    ys = np.array([np.sin(x+ai) for ai in a])
    #ys = np.array([ai*x**3 for ai in a])
    return x, ys #a * xs**2


#def integration_weights_(x):
#    return 0.5 * np.concatenate((np.array([x[1] - x[0]]),
#                              x[2:] - x[:(len(x) - 2)],
#                              np.array([x[len(x) - 1] - x[len(x) - 2]]))
#                              ,axis=None)
#
#def eig(X,Y):
#    n_obs = X.size
#    argvals = X.copy()
#    data = Y.copy()
#    weight = integration_weights_(argvals)
#    #print(weight)
#    # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
#    print(weight)
#    weight_sqrt = np.diag(np.sqrt(weight))
#    weight_invsqrt = np.diag(1 / np.sqrt(weight))
#    mean = Y.mean(axis=0)
#    data -= mean
#    covariance = data.T.dot(data) / (n_obs-1)
#    covariance += covariance.T
#    covariance /= 2
#    var = np.dot(np.dot(weight_sqrt, covariance), weight_sqrt)
#
#    eigenvalues, eigenvectors = np.linalg.eigh(var)
#    eigenvalues[eigenvalues < 0] = 0
#    eigenvalues = eigenvalues[::-1]
#    # Slice eigenvalues and compute eigenfunctions = W^{-1/2}U
#    exp_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
#    npc = np.sum(exp_variance < 0.99) + 1
#
#    eigenvalues = eigenvalues[:npc]
#    eigenfunctions = np.transpose(np.dot(weight_invsqrt,
#                                         np.fliplr(eigenvectors)[:, :npc]))
#    return eigenvalues, eigenfunctions, npc
#
#if __name__ == "__main__":
#    num_func = 100
#    np.random.seed(24)
#
#    dense = 1000
#
#    #x = np.linspace(-5,5, dense); X,Y = foo(x);
#    x = np.sort(np.random.uniform(-5,5, dense)); X,Y = foo(x);
#
#    print(X,Y)
#
#    eigenvalues, eigenfunctions, npc = eig(X,Y)
#    [plt.plot(X, eigenfunctions[i]) for i in range(npc)]
#    plt.show()


import pandas as pd

from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.misc.loader import read_csv
from FDApy.misc.utils import integration_weights_

#temperature = read_csv('/home/taylanot/Desktop/sandbox/FDApy/examples/data/canadian_temperature_daily.csv', index_col=0)

def pca(num_func = 1000):
    dense=1000
    #x = np.sort(np.random.normal(0,1, dense)); X,Y = foo(x, num_func,);
    #x = np.sort(np.random.uniform(-3,3, dense)); X,Y = foo(x, num_func,);
    x = np.sort(np.linspace(-5,5, dense)); X,Y = foo(x, num_func,);
    data = DenseFunctionalData({'input_dim_0':X}, Y)


    # Perform a univariate FPCA on dailyTemp.
    fpca = UFPCA(n_components=0.99)
    fpca.fit(data)

    # Plot the results of the FPCA (eigenfunctions)
    [plt.plot(X, Y[i], alpha=0.1) for i in range(num_func)]
    #print(fpca.eigenvalues)
    #_ = plot(fpca.eigenfunctions)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Functional PCA 99% variance")
    directory = "functional_pca_res"
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "illustration-2.pdf")
    xnew = np.linspace(-5,5,100)
    ncomps = fpca.eigenfunctions.values.shape[0]
    for i in range(ncomps): 
        ynew, smooth = csaps(x,fpca.eigenfunctions.values[i],xnew)
        plt.plot(xnew, ynew, '-')

    plt.show()
    #plt.savefig(path)

if __name__ == "__main__":
    np.random.seed(24)
    pca(1000)
