from fenics_.src.genrve import *


hs = np.array([1,5,10,50,100])/100
Vfs  = np.array([1,2,3,4])/10
generator = GENERATE_RVE()

for h in hs:
    for Vf in Vfs:
    generator(Vf, h)


