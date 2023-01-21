import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0,100)
def POW2(n, a=1, bs=np.linspace(0,2), c=0):
    return [a*n**(-b) for b in bs]

def LOG2(n=np.linspace(0,100), a=1, b=np.linspace(-2,2), c=0):
    return -a*np.log(n) + c

print(POW2(n))
data = POW2(n)
[plt.plot(n, data[i]) for i in range(50)]

plt.show()
