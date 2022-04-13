import numpy as np
import matplotlib.pyplot as plt

set1 = np.genfromtxt('retreat2022/set1.dat',delimiter=',')
set2 = np.genfromtxt('retreat2022/set2.dat',delimiter=',')

def info(dataset):
    print('# of objects:{}'.format(dataset.shape[0]))
    print('# of features:{}'.format(dataset.shape[1]-1))
    print('var of features:{}'.format(np.var(dataset[0:-1],axis=0)))
    print('mean of features:{}'.format(np.mean(dataset[0:-1],axis=0)))

#info(set1)
#info(set2)


