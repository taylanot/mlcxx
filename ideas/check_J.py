import numpy as np
import copy as cp 
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

np.random.seed(24)

def func(x, a):
    return  a @ (x**2)

def data(D=2,N=10, noise=False):
    xi =  np.linspace(1,10,N)
    x =  np.vstack((xi,xi))
    y =  func(x,np.ones(D)) 
    if noise:
        y += np.random.normal(0,0.1,N)
    return x, y

def lm_func(t,p):
    """
    Define model function used for nonlinear least squares curve-fitting.
    Parameters
    ----------
    t     : independent variable values (assumed to be error-free) (m x 1)
    p     : parameter values , n = 4 in these examples             (n x 1)
    Returns
    -------
    y_hat : curve-fit fctn evaluated at points t and with parameters p (m x 1)
    """
    
    #y_hat = p[0,0]*np.exp(-t/p[1,0]) + p[2,0]*np.sin(t/p[3,0])
    
    return func(t,p)


def lm_FD_J(t,p,y,dp):
    """
    Computes partial derivates (Jacobian) dy/dp via finite differences.
    Parameters
    ----------
    t  :     independent variables used as arg to lm_func (m x 1) 
    p  :     current parameter values (n x 1)
    y  :     func(t,p,c) initialised by user before each call to lm_FD_J (m x 1)
    dp :     fractional increment of p for numerical derivatives
                - dp(j)>0 central differences calculated
                - dp(j)<0 one sided differences calculated
                - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed
    Returns
    -------
    J :      Jacobian Matrix (n x m)
    """

    
    # number of data points
    m = len(y)
    # number of parameters
    n = len(p)

    # initialize Jacobian to Zero
    ps=cp.deepcopy(p)
    J=np.zeros((m,n)) 
    del_=np.zeros((n,1))
    # START --- loop over all parameters
    for j in range(n):
        # parameter perturbation
        del_[j,0] = dp * (1+abs(p[j]))
        # perturb parameter p(j)
        p[j]   = ps[j] + del_[j,0]
        
        if del_[j,0] != 0:
            y1 = lm_func(t,p)
            
            if dp < 0: 

                print(p)
                # backwards difference
                J[:,j] = (y1-y)/del_[j,0]
            else:
                # central difference, additional func call
                p[j] = ps[j] - del_[j]
                J[:,j] = (y1-lm_func(t,p)) / (2 * del_[j,0])
        # restore p(j)
        p[j]=ps[j]
    return J

D = 2 
N = 10
noise = False
x, y = data(D, N, noise)
p = np.ones(D)
dp = -0.001;
print(lm_FD_J(x, p, y, dp))
