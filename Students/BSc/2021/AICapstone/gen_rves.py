from fenics_.src.genrve import *

Vfs  = np.array([1,2,3,4])/10
num = 100
generator = CreateRVEGeometry(name="Gallery")

for Vf in Vfs:
    for i in range(num):
        generator(Vf,tag=i+1)


