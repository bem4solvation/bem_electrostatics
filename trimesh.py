import time
import numpy 
import trimesh


def fix_mesh(mesh):

    mesh.fill_holes()
    mesh.process()
    iter_limit = 20
    
    iteration = 0
    while not mesh.is_watertight and iteration<iter_limit:
        merge_tolerance = 0.05
        needy_faces     = trimesh.repair.broken_faces(mesh)
        for vert_nf in mesh.faces[needy_faces]:
            for nf in vert_nf:
                for c,check in enumerate(numpy.linalg.norm(mesh.vertices[vert_nf]-mesh.vertices[nf],axis=1)):
                    if (check<merge_tolerance) & (0<check):
                        mesh.vertices[nf]=mesh.vertices[vert_nf[c]]
        iteration += 1
    
    if iteration>iter_limit-1: 
        print (' not watertight') 
                        
    mesh.fill_holes()
    mesh.process()
    
    return mesh 