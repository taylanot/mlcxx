import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cycler
import tikzplotlib

def lin(a, x):
    return a*x

def nonlin(a1,a2, x):
    return a1*np.sin(a2*x)


n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

fig, ax = plt.subplots(figsize=(6,6))

xlin = np.linspace(-1,1,100).reshape(-1,1)
xnonlin = np.linspace(-5,5,100).reshape(-1,1)

for i in range(100):
    y = lin(np.random.normal(0,1), xlin)
    ax.plot(xlin, y)
tikzplotlib.save("lin_eg.tex")

plt.close()
fig2, ax2 = plt.subplots(figsize=(6,6))


for i in range(100):
    y = nonlin(np.random.normal(1,1),np.random.normal(0,1), xnonlin)
    ax2.plot(xnonlin, y)
tikzplotlib.save("nonlin_eg.tex")


