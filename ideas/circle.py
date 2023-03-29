import numpy as np 
import matplotlib.pyplot as plt 
import math
#
#
#num_samples = 1000
#N = num_samples
#
## make a simple unit circle 
#theta = np.linspace(0, 2*np.pi, num_samples)
#a, b = 1 * np.cos(theta), 1 * np.sin(theta)
#x, y = a+np.random.normal(0,0.1,N), b+np.random.normal(0,0.1,N)
#
#
## generate the points
## theta = np.random.rand((num_samples)) * (2 * np.pi)
#r = np.random.rand((num_samples))
##x, y = r * np.cos(theta), r * np.sin(theta)
#
## plots
#plt.figure(figsize=(7,6))
#plt.plot(a, b, linestyle='-', linewidth=2, label='Circle')
#plt.scatter(x, y)
#plt.ylim([-1.5,1.5])
#plt.xlim([-1.5,1.5])
#plt.grid()
#plt.legend(loc='upper right')
#plt.show(block=True)
#
#import random
#import math

#Get a random point on a unit hypersphere of dimension n
#def random_hypersphere_point(n):
#    #fill a list of n uniform random values
#    #points = np.array([np.random.normal(0, 4.0) for r in range(n)])
#    points = np.array([np.random.normal(0, 4.0) for r in range(n)])
#    #calculate 1 / sqrt of sum of squares
#    sqr_red = 1.0 / math.sqrt(sum(i*i for i in points))
#    #multiply each point by scale factor 1 / sqrt of sum of squares
#    #return map(lambda x: x * sqr_red, points)
#    return points * sqr_red
#
#points = list() 
#for i in range(1000):
#    points.append(random_hypersphere_point(2))
#points = np.array(points)
###points += np.random.normal(0,0.1, points.shape)
def dip(N,D):
    X = np.random.normal(0,10,(D,N));
    return X / np.sqrt((X**2).sum(axis=0))
    
points = dip(1000, 3);
bounds = [[0,10],[0,10],[0,10]]
r =  10

points *= r
points += np.random.normal(0,0.1, points.shape)
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(points[0,:],points[1,:], points[2,:])
op = np.random.multivariate_normal(np.zeros(3), np.eye(3), (3,1000))
ax.scatter3D(op[0,:],op[1,:], op[2,:])
plt.show()

