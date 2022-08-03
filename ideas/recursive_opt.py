from scipy.optimize import minimize
from scipy import optimize
import numpy as np

np.random.seed(24)

#def f(x):
#    return (x[0]**2 + x[1]**2 - 1)**2
#
#bounds = [(-10,10),(-10,10)]
#
#results = dict()
#results['shgo'] = optimize.shgo(f, bounds)
#results['shgo_sobol'] = optimize.shgo(f, bounds, n=1000, iters=5,
#                                      sampling_method='sobol')
#print(results['shgo_sobol']['x']) # ->  0.3851043, 0.6148957
#
#
#def df_dx(x):
#    return 4*x[0]*(x[0]**2+x[1]**2-1), 4*x[1]*(x[0]**2+x[1]**2-1)
#
#x = np.random.uniform(0,1,(2))
#lr = 1e-3
#for i in range(1000000):
#    grad = df_dx(x) 
#    x[0] -= lr*grad[0]
#    x[1] -= lr*grad[1]
#
#print(x)

#def f(x):
#    return (x[0] + x[1] - 1)**2
#
#bounds = [(-10,10),(-10,10)]
#
#results = dict()
#results['shgo'] = optimize.shgo(f, bounds)
#results['shgo_sobol'] = optimize.shgo(f, bounds, n=1000, iters=5,
#                                      sampling_method='sobol')
#print(results['shgo_sobol']['x']) # ->  0.3851043, 0.6148957

#def solve():
#    X = np.ones((2,2))
#    y = np.ones((2,1))
#    A = np.linalg.lstsq(X,y)
#    return A
#
#print(solve())

# NOT A GOOD IDEA ABORT ABORT ABORTTTTTT !!!! 


#x = 100.
#
#y = 100.
#
#for i in range(100):
#    x = -y/2
#    y = -x/2
#
#print(x,y)

psi = [np.sin, np.cos]
x = np.random.uniform(0,1,10).reshape(-1,1)
Psi = np.hstack(([psi[i](x) for i in range(len(psi))]))
print(np.linalg.pinv(Psi))

