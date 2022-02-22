import meshio
import numpy as np
import sys
import os

def write_2D(filename):
    head, tail = os.path.split(filename)
    if tail == 'rve.msh':
        mesh = meshio.read(filename)
        a_file = open(head+'/rve_jive.msh', 'w')
        a_file.write("$MeshFormat\n")
        a_file.write("2.2 0 8\n")
        a_file.write("$EndMeshFormat\n")
        a_file.write("$Nodes\n")
        nodes(a_file, mesh)
        a_file.write("$EndNodes\n")
        a_file.write("$Elements\n")
        a_file.write(str(mesh.cells_dict['triangle'].shape[0])+'\n')
        elements(a_file,mesh)
        a_file.write("$EndElements\n")
        a_file.close()

def nodes(filename,mesh):
    data = np.hstack(((np.arange(mesh.points.shape[0])+1).reshape(-1,1),mesh.points))
    filename.write(str(data.shape[0])+'\n')
    np.savetxt(filename, data ,fmt='%i %e %e %i')

def elements(filename,mesh):
    cells = mesh.cells_dict['triangle']+1
    geo_tags = mesh.cell_data_dict['gmsh:geometrical']['triangle'].reshape(-1,1)-1
    phys_tags = mesh.cell_data_dict['gmsh:physical']['triangle'].reshape(-1,1)-1
    tags = np.ones((cells.shape[0],2))*2
    data = np.hstack(((np.arange(cells.shape[0])+1).reshape(-1,1),tags,phys_tags,geo_tags,cells))
    np.savetxt(filename, data ,fmt='%i')
    
write_2D(filename=sys.argv[1])
#write_2D('rve.msh')
