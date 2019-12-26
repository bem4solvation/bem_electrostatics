import numpy as np
import subprocess
import os
import bempp.api

def convert_pdb2pqr(mesh_pdb_path, mesh_pqr_path):       
    subprocess.call(["python", "/home/chris/Software/apbs-pdb2pqr/pdb2pqr/pdb2pqr.py", "--ff=amber", mesh_pdb_path, mesh_pqr_path])
    
def convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path):
    pqr_file = open(mesh_pqr_path, 'r')
    pqr_data = pqr_file.read().split('\n')
    xyzr_file = open(mesh_xyzr_path, 'w')
    for line in pqr_data:
        line = line.split()
        if len(line)==0 or line[0]!='ATOM': continue
        xyzr_file.write(line[5]+"\t"+line[6]+"\t"+line[7]+"\t"+line[9]+"\n")
    pqr_file.close()
    xyzr_file.close()
    
def convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path):
    face = open(mesh_face_path,'r').read()
    vert = open(mesh_vert_path,'r').read()

    faces = np.vstack(np.char.split(face.split('\n')[0:-1]))[:,:3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split('\n')[0:-1]))[:,:3].astype(float)

    data = open(mesh_off_path, 'w')
    data.write("OFF"+"\n")
    data.write(str(verts.shape[0])+"\t"+str(faces.shape[0])+"\n")
    for vert in verts:
        data.write(str(vert[0])+"\t"+str(vert[1])+"\t"+str(vert[2])+"\n")
    for face in faces:
        data.write("3"+"\t"+str(face[0])+"\t"+str(face[1])+"\t"+str(face[2])+"\n")
    
def generate_msms_mesh(mesh_xyzr_path, output_dir, output_name, density, probeRadius):
    command = "msms -if "+mesh_xyzr_path+" -of "+output_dir+output_name+" -p "+str(probeRadius)+" -d "+str(density)+" -no_header"
    os.system(command)
    
def import_msms_mesh(mesh_face_path, mesh_vert_path):
    face = open(mesh_face_path,'r').read()
    vert = open(mesh_vert_path,'r').read()

    faces = np.vstack(np.char.split(face.split('\n')[0:-1]))[:,:3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split('\n')[0:-1]))[:,:3].astype(float)

    grid = bempp.api.grid_from_element_data(verts.transpose(), faces.transpose())
    return grid