from fenics_.src.genrve import *

#Vfs  = np.array([1,2,3,4])/10
#num = 100
#generator = CreateRVEGeometry(name="Gallery", ext='.geo_unrolled')
#
#for Vf in Vfs:
#    for i in range(num):
#        generator(Vf,tag=i+1)

generator = GENERATE_RVE(name='Gallery',ext='.msh2')
Vfs  = np.array([1,2,3,4])/10
hs  = np.array([0.1,0.2,0.5,1])

num = 100
for Vf in Vfs:
    for i in range(num):
        for h in hs:
            np.random.seed(i)
            generator.mesh_size=h
            generator(Vf, tag=i+1)


