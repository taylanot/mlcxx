#
# @file plotter.py
# @author Ozgur Taylan Turan
#
# Simple plotting tools for the output of mlcxx
#
# TODO: Prediction Plots with Train and Test 
#
#
# Plotting Stuff
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cycler
import numpy as np
import tikzplotlib
from read import * 
# Define your plot cycle color
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',
                                            plt.cm.Dark2(np.linspace(0, 1, 8)))

def lcurve(ax, data, error=True, dots=True):
    if error:
        ax.fill_between(data[:,0],
                        data[:,1] - data[:,2],
                        data[:,1] + data[:,2],
                        alpha=0.1)
        ax.fill_between(data[:,0],
                        data[:,3] - data[:,4],
                        data[:,3] + data[:,4],
                        alpha=0.1)
    if dots: 
        ax.plot(data[:,0], data[:,1], '-o', linewidth=0.2, label="Train-Error")
        ax.plot(data[:,0], data[:,3], '-o', linewidth=0.2, label="Test-Error")
    else:
        ax.plot(data[:,0], data[:,1], linewidth=0.2, label="Train-Error")
        ax.plot(data[:,0], data[:,3], linewidth=0.2, label="Test-Error")

def data(ax,data,din,dout, label='Train'):
    if label == "Train":
        ax.plot(data[:,0:din], data[:,-dout:],'x', markerfacecolor=None, label=label)
    else:
        ax.plot(data[:,0:din], data[:,-dout:], 'x', markerfacecolor=None, label=label)

def pred(ax,data,din,dout):
    ax.plot(data[:,0:din], data[:,-dout:], '-o', label="Prediction")

