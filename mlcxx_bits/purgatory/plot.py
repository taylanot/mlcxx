from numpy import genfromtxt
import matplotlib.pyplot as plt

xtrn = genfromtxt('inputs.csv', delimiter=',')
ytrn = genfromtxt('labels.csv', delimiter=',')

xtst= genfromtxt('pred_inputs.csv', delimiter=',')
ytst= genfromtxt('pred_labels.csv', delimiter=',')
plt.scatter(xtrn, ytrn)
plt.plot(xtst, ytst)
plt.show()
