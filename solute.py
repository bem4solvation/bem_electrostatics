import re
import bempp.api
import os
import numpy as np
import bem_electrostatics.mesh_tools.mesh_tools as mesh_tools
import bem_electrostatics.utils as utils
import bem_electrostatics.pb_formulation as pb_formulation

class solute():
    """The basic Solute object

    This object holds all the solute information and allows for a easy way to hold the data"""

    def __init__(self, solute_file_path, save_mesh_build_files = False, mesh_build_files_dir = "mesh_files/", mesh_density = 2, mesh_probe_radius = 1.4):

        if os.path.isfile(solute_file_path) == False:
            print("file does not exist -> Cannot start")
            return

        self.save_mesh_build_files = save_mesh_build_files
        self.mesh_build_files_dir = mesh_build_files_dir
        self.mesh_density = mesh_density
        self.mesh_probe_radius = mesh_probe_radius

        file_extention = solute_file_path.split(".")[-1]
        if file_extention == "pdb":
            self.imported_file_type = "pdb"
            self.pdb_path = solute_file_path
            self.solute_name = get_name_from_pdb(self.pdb_path)

        elif file_extention == "pqr":
            self.imported_file_type = "pqr"
            self.pqr_path = solute_file_path
            self.solute_name = solute_file_path.split(".")[-2].split("/")[-1]

        else:
            print("File is not pdb or pqr -> Cannot start")



        self.mesh, self.q, self.x_q = generate_msms_mesh_import_charges(self)
        self.pb_formulation = "direct"

        self.ep_in = 1.0
        self.ep_ex = 80.0
        self.kappa = 0.125

        self.pb_formulation_alpha = 1.0
        self.pb_formulation_beta = self.ep_ex/self.ep_in
        
        self.gmres_tolerance = 1e-5
        self.gmres_max_iterations = 1000
                



    def calculate_potential(self):
        dirichl_space = bempp.api.function_space(self.mesh, "P", 1)
        neumann_space = bempp.api.function_space(self.mesh, "P", 1)

        self.dirichl_space = dirichl_space
        self.neumann_space = neumann_space

        if self.pb_formulation == "juffer":
            A, rhs = pb_formulation.juffer(dirichl_space, neumann_space, self.q, self.x_q, self.ep_in, self.ep_ex, self.kappa)
        elif self.pb_formulation == "direct":
            A, rhs = pb_formulation.direct(dirichl_space, neumann_space, self.q, self.x_q, self.ep_in, self.ep_ex, self.kappa)
        elif self.pb_formulation == "alpha_beta":
            A, rhs = pb_formulation.alpha_beta(dirichl_space, neumann_space, self.q, self.x_q, self.ep_in, self.ep_ex, self.kappa, self.pb_formulation_alpha, self.pb_formulation_beta)

        x, info, it_count = utils.solver(A, rhs, self.gmres_tolerance, self.gmres_max_iterations)

        self.solver_iteration_count = it_count
        self.phi = x[:dirichl_space.global_dof_count]
        self.d_phi = x[dirichl_space.global_dof_count:]



    def calculate_solvation_energy(self):
        if not hasattr(self, 'phi'):
            #print("You must first calulate the potential (solute.calculate_potential)")
            #call calulate potential here
            self.calculate_potential()

        dirichl_space = self.dirichl_space
        neumann_space = self.neumann_space

        solution_dirichl = bempp.api.GridFunction(dirichl_space, coefficients=self.phi)
        solution_neumann = bempp.api.GridFunction(neumann_space, coefficients=self.d_phi)

        from bempp.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(neumann_space, self.x_q.transpose())
        dlp_q = double_layer(dirichl_space, self.x_q.transpose())
        phi_q = slp_q*solution_neumann - dlp_q*solution_dirichl

        # total solvation energy applying constant to get units [kcal/mol]
        total_energy = 2*np.pi*332.064*np.sum(self.q*phi_q).real

        self.solvation_energy = total_energy



    def mesh_info(self):
        print("The grid has:")

        number_of_elements = self.mesh.leaf_view.entity_count(0)
        print("{0} elements".format(number_of_elements))




def generate_msms_mesh_import_charges(solute):

    mesh_dir = "mesh_temp/"
    if solute.save_mesh_build_files:
        mesh_dir = solute.mesh_build_files_dir

    if not os.path.exists(mesh_dir):
        try:
            os.mkdir(mesh_dir)
        except OSError:
            print ("Creation of the directory %s failed" % mesh_dir)

    if solute.imported_file_type == "pdb":
        mesh_pqr_path = mesh_dir+solute.solute_name+".pqr"
        mesh_tools.convert_pdb2pqr(solute.pdb_path, mesh_pqr_path)
    else:
        mesh_pqr_path = solute.pqr_path

    mesh_xyzr_path = mesh_dir+solute.solute_name+".xyzr"
    mesh_tools.convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path)

    mesh_face_path = mesh_dir+solute.solute_name+".face"
    mesh_vert_path = mesh_dir+solute.solute_name+".vert"
    mesh_tools.generate_msms_mesh(mesh_xyzr_path, mesh_dir, solute.solute_name, solute.mesh_density, solute.mesh_probe_radius)

    grid = mesh_tools.import_msms_mesh(mesh_face_path, mesh_vert_path)
    q, x_q = utils.import_charges(mesh_pqr_path)

    if solute.save_mesh_build_files:
        if solute.imported_file_type == "pdb":
            solute.mesh_pqr_path = mesh_pqr_path
        solute.mesh_xyzr_path = mesh_xyzr_path
        solute.mesh_face_path = mesh_face_path
        solute.mesh_vert_path = mesh_vert_path
    else:
        if solute.imported_file_type == "pdb":
            os.remove(mesh_pqr_path)
        os.remove(mesh_xyzr_path)
        os.remove(mesh_face_path)
        os.remove(mesh_vert_path)
        os.rmdir(mesh_dir)

    return grid, q, x_q

def get_name_from_pdb(pdb_path):
    pdb_file = open(pdb_path)
    firstline = pdb_file.readline()
    firstline_split = re.split(r'\s{2,}', firstline)
    solute_name = firstline_split[3].lower()
    pdb_file.close()

    return solute_name
