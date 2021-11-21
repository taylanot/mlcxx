import numpy as np
import GPy.models as GPy
from sklearn.cluster import KMeans
import scipy.spatial as ss
import matplotlib.pyplot as plt

# Weighing is done with this function
def weights(x, c, l):
    num     =   x.shape[0]
    dim     =   x.shape[1]     
    numc    =   c.shape[0] 
    distances   =   np.zeros((num,numc))
    for i in range (0,numc):
        for j in range (0,num):
            distances[j,i]   =   np.exp(-0.5*np.dot((x[j,:]-c[i,:]),(x[j,:]-c[i,:]).T)/l[i]**2)
    print(distances)
    tot =   np.sum(distances,axis=1)
    distances = distances/tot.reshape(-1,1)
    return distances 

# Simple center calculation for 1D but, can use cluster centers from sklearn
def center(points):
    centroid  = np.sum(points,axis=0) / points.shape[0] 
    return centroid

# Initialize some parameters
np.random.seed(81)
num     =   10
numm    =   20 
noise   =   np.random.uniform(0,0.4,(num,1))
noisem  =   np.random.uniform(0,0.2,(numm,1))

#X1  =   np.random.uniform(0,6.,(num,1));    y1  =   np.sin(X1) + noise 
#X2  =   np.random.uniform(9,14,(numm,1));    y2  =   np.sin(2*X2) + noisem

# Generate data and fit GPs on different domains
X1  =   np.linspace(0,6,10).reshape(-1,1);    y1  =   np.sin(X1) + noise 
X2  =   np.linspace(7,14,20).reshape(-1,1);    y2  =   np.sin(4*X2) + noisem 
m1  =   GPy.GPRegression(X1,y1);    m1.optimize_restarts(num_restarts=4)
m2  =   GPy.GPRegression(X2,y2);    m2.optimize_restarts(num_restarts=4)

fig, ax =   plt.subplots()

m1.plot_mean(ax=ax,color='orange', label="GP1-mean")
m1.plot_confidence(ax=ax,color='orange', label="GP1-conf")
m2.plot_mean(ax=ax, label='GP2-mean')
m2.plot_confidence(ax=ax,label='GP2-conf')

plt.xlim([0,14])

X   =   np.vstack((X1,X2))
y   =   np.vstack((y1,y2))

# Cluster 
kmeans  =   KMeans(n_clusters=2)
kmeans.fit(X)
ax.scatter(X,y,c=kmeans.labels_,cmap='rainbow', facecolors='none',label='data')
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("local.pdf")

# Complete GP in the domain
m   =   GPy.GPRegression(X,y); m.optimize_restarts(num_restarts=4)
fig1, ax1   =   plt.subplots()
m.plot(ax=ax1)
plt.xlim([0,14])

# Get the lengthscale
l   = [m1['rbf.lengthscale'],m2['rbf.lengthscale']]
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("global.pdf")

# Cluster centers
figc, axc   =   plt.subplots()
#c   =   kmeans.cluster_centers_
c1  =   center(X1)
c2  =   center(X2)
c   =   np.vstack((c1,c2))
x   =   np.linspace(0,14,1001).reshape(-1,1)
pi  =   weights(x,c,l)

# Weigh predictions
mean1, var1     =   m1.predict(x)
mean2, var2     =   m2.predict(x)
mean    =   mean1*pi[:,0].reshape(-1,1) + mean2*pi[:,1].reshape(-1,1)
var     =   var1*pi[:,0].reshape(-1,1) + var2*pi[:,1].reshape(-1,1)

# Plot EoM Results
axc.plot(x,mean,label='mean')
axc.fill_between(x.ravel(),(mean+np.sqrt(var)).ravel(),(mean-np.sqrt(var)).ravel(),alpha=0.4,label='conf')
axc.scatter(X,y,c=kmeans.labels_,cmap='rainbow', facecolors='none',label='data')
axc.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("cluster.pdf")
