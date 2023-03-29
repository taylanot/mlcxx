import numpy as np
from scipy.stats import wilcoxon, f_oneway

def test(N):
    #return wilcoxon(np.random.uniform(-1,1,N),np.random.normal(0,1,N))
    #return wilcoxon(np.random.uniform(-1,1,N),np.random.normal(0,1,N))
    return f_oneway(np.random.normal(0.,1.,N),np.random.normal(0.,0.001,N))

print(test(10))
