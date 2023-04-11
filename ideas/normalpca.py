import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def func(x, noise=False):
    phi = np.random.uniform(0,2*np.pi);
    if noise:
        return np.sin(x+phi) + np.random.normal(0,0.1)
    else:
        return np.sin(x+phi)

x = np.linspace(0,2,1000);
noise = True;
M = 100;
ys = [func(x,noise) for i in range(M)];
y = np.vstack(ys).T

npc = 100
pca = PCA(npc);
yp = pca.fit_transform(y)
print(yp)
[plt.scatter(x,yp[:,i]) for i in range(1)];
plt.show()


