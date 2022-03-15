# File Handling Stuff
import json
import os
from glob import glob
# Stats from statistics import mean, pstdev
# Plotting Stuff
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cycler
import numpy as np
import os
# Define your plot cycle color
n = 8
color = plt.cm.Dark2(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

Experiment_Directory = 'unimodal-nonlinear/MAML/std_y/1'
run_file = os.path.join(Experiment_Directory, 'run.json')
with open(run_file) as json_file:
    run_info = json.load(json_file)
print(run_info)

#res = run_info['result'][0]

#plt.plot(res[0:20])
#
#Experiment_Directory = 'unimodal-nonlinear/Bayes/std_y/1'
#run_file = os.path.join(Experiment_Directory, 'run.json')
#with open(run_file) as json_file:
#    run_info = json.load(json_file)
#res = run_info['result'][0]
#plt.plot(res[0:20])
#
#plt.savefig('check.pdf')
