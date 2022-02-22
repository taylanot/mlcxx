import meshio
import numpy as np
import sys
import os

def write_2D(step_file='A.step', h=0.5):
    head, tail = os.path.split(step_file)
    os.system('/home/shared/Tools/gmsh/bin/gmsh '+step_file+' -2 -o '+head+'/rve.msh -format msh22 -clmax '+str(h)) 
#    mesh = meshio.read(head+'/rve.msh')
#    num_ele = mesh.cells_dict['triangle'].shape[0]
#    a_file = open(head+'/rve_jive_'+str(num_ele)+'.msh', 'w')
#    a_file.write("$MeshFormat\n")
#    a_file.write("2.2 0 8\n")
#    a_file.write("$EndMeshFormat\n")
#    a_file.write("$Nodes\n")
#    nodes(a_file, mesh)
#    a_file.write("$EndNodes\n")
#    a_file.write("$Elements\n")
#    a_file.write(str(mesh.cells_dict['triangle'].shape[0])+'\n')
#    elements(a_file,mesh)
#    a_file.write("$EndElements\n")
#    a_file.close()
#
#def nodes(filename,mesh):
#    data = np.hstack(((np.arange(mesh.points.shape[0])+1).reshape(-1,1),mesh.points))
#    filename.write(str(data.shape[0])+'\n')
#    np.savetxt(filename, data ,fmt='%i %e %e %i')
#
#def elements(filename,mesh):
#    cells = mesh.cells_dict['triangle']+1
#    cells_data = mesh.cell_data_dict['gmsh:geometrical']['triangle'].reshape(-1,1)-1
#    geo_tags = np.ones((cells.shape[0],2))*2
#    phys_tags = np.zeros((cells.shape[0],1))
#    data = np.hstack(((np.arange(cells.shape[0])+1).reshape(-1,1),geo_tags,phys_tags,cells_data,cells))
#    np.savetxt(filename, data ,fmt='%i')
    
write_2D(step_file=sys.argv[1],h=sys.argv[2])

