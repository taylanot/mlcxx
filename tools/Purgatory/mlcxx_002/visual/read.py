#
# @file read.py
# @author Ozgur Taylan Turan
#
# Simple functions for reading data from different file types
#
# TODO: Prediction Plots with Train and Test 
#
#



# Reader Stuff
import numpy as np
import os

def load_csv(filename,order):
    if order != "row":
        return np.genfromtxt(filename, delimiter=',').t()
    else:
        return np.genfromtxt(filename, delimiter=',')



