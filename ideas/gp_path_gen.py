import GPy
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(8)
#Training data
#Input = np.linspace(0,100,11).reshape(-1,1)
a = np.array([0.75, 0.5, -0.75, 0., -0.5])
#Output = np.hstack((np.zeros(1), np.random.choice(a, len(Input)-1))).reshape(-1,1)
#Output = 0.1 * np.sin(Input)


def generate_path(scale=1):
    dim = 1
    Input = np.arange(0,101, 20).reshape(-1,1)
    Output = np.random.uniform(-0.75, 0.75, size=(len(Input),1))
    #Output = np.random.choice(a, len(Input)).reshape(-1,1)
    Output[0,0] = 0
    #Output[-1,-1] = 1
    #print(Output)
    #Build Gaussian process regression model
    kernel = GPy.kern.RBF(1, variance=1e-12)
    model = GPy.models.GPRegression(Input, Output, kernel)
    model['Gaussian_noise.variance'].fix(1.e-6)
    model['rbf.lengthscale'].fix(25)
    #model.optimize_restarts(num_restarts=10)
    model.optimize()
    testX = np.arange(0, 101).reshape(-1, 1)
    mean, var = model.predict(testX)
    N = 1 
    posterior_eval = model.posterior_samples_f(testX, full_cov=True, size=N)
    #simY, simMse = model.predict(testX)
    #for i in range(N):
    #    plt.plot(testX, scale*posterior_eval[:,0,i])
    #plt.scatter(Input, scale*Output)
    return posterior_eval
    #drawing
    #model.plot()
    #plt.gcf().set_size_inches(8, 8, forward=True)
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    #plt.xlabel("x1")
    #plt.ylabel("x2")
N = 1000 
paths = np.array([np.vstack((generate_path()[:,0,0],generate_path()[:,0,0])) for i in range(N)])
np.save('path_collection.npy',paths)
#plt.show()
