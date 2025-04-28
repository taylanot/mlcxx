import numpy as np
import matplotlib.pyplot as plt
import polars as ps
from dtw import *
# whole_cos = np.genfromtxt("whole_cos.csv",delimiter=",")
# part_cos = np.genfromtxt("whole_cos.csv",delimiter=",")

cos = np.genfromtxt("cos_curves.csv",delimiter=",")
curve_ = np.genfromtxt("curve_curves.csv",delimiter=",")


curves = ps.read_csv(".llc-paper/LCDB_0_12/classification/train.csv").to_numpy().T
anchors = ps.read_csv(".llc-paper/LCDB_0_12/classification/test.csv").to_numpy().T
curves = curves[1:]
anchor = anchors[1]

# dists = []
# for curve in curves:
#     dists.append(dtw(curve, anchor, keep_internals=True).distance)
# dists = np.array(dists)

# idx = np.argsort(dists)

till = 10

plt.plot(cos[0],'g', label="anchor")
for i in range(1,till):
    plt.plot(cos[i],'r', label="cos")
    #plt.plot(curve_[i],'b--', label="mijn")
    #plt.plot(curves[idx[i]+1],'y:', label="dtw")

plt.legend()
plt.show()


