import re
import bempp.api
import os
import numpy as np
import time
import bem_electrostatics.mesh_tools.mesh_tools as mesh_tools
import bem_electrostatics.utils as utils
import bem_electrostatics.pb_formulation as pb_formulation

class solute():
    """The basic Solute object

    This object holds all the solute information and allows for a easy way to hold the data"""

    def __init__(self, solute_file_path, external_mesh_file = None, save_mesh_build_files = False, mesh_build_files_dir = "mesh_files/", mesh_density = 1.0, mesh_probe_radius = 1.4, mesh_generator = "nanoshaper", print_times = False, force_field = "amber"):

        if os.path.isfile(solute_file_path) == False:
            print("file does not exist -> Cannot start")
            return
        
        self.force_field = force_field

        self.save_mesh_build_files = save_mesh_build_files
        self.mesh_build_files_dir = os.path.abspath(mesh_build_files_dir)
        self.mesh_density = mesh_density
        self.mesh_probe_radius = mesh_probe_radius
        self.mesh_generator = mesh_generator
        
        self.print_times = print_times

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
            
        if external_mesh_file is not None:
            filename, file_extension = os.path.splitext(external_mesh_file)
            if file_extension == "":  ## Assume use of vert and face
                self.external_mesh_face_path = external_mesh_file+".face"
                self.external_mesh_vert_path = external_mesh_file+".vert"
                self.mesh = mesh_tools.import_msms_mesh(self.external_mesh_face_path, self.external_mesh_vert_path)
                
            else:  ## Assume use of file that can be directly imported into bempp
                self.external_mesh_file_path = external_mesh_file
                self.mesh = bempp.api.import_grid(self.external_mesh_file_path)
            
            self.q, self.x_q = import_charges(self)  ## Import charges from given file
            
        else: ## Generate mesh from given pdb or pqr, and import charges at the same time
            self.mesh, self.q, self.x_q = generate_msms_mesh_import_charges(self)


        self.mesh_elements = self.mesh.number_of_elements
        self.pb_formulation = "direct"

        self.ep_in = 4.0
        self.ep_ex = 80.0
        self.kappa = 0.125

        self.pb_formulation_alpha = 1.0
        self.pb_formulation_beta = self.ep_ex/self.ep_in
        
        self.pb_formulation_preconditioning = False
        self.pb_formulation_preconditioning_type = "squared"
        
        self.discrete_form_type = "strong"
        
        self.gmres_tolerance = 1e-5
        self.gmres_max_iterations = 1000
        
        #bempp.api.set_default_device(0,0)
        #print(bempp.api.default_device())
                


    def calculate_potential(self):
        ## Start the overall timing for the whole process
        start_time = time.time()
        
        ## Setup Dirichlet and Neumann spaces to use, save these as object vars ##
        dirichl_space = bempp.api.function_space(self.mesh, "P", 1)
        neumann_space = bempp.api.function_space(self.mesh, "P", 1)
        self.dirichl_space = dirichl_space
        self.neumann_space = neumann_space
        
        ## Construct matrices and rhs based on the desired formulation ##
        setup_start_time = time.time() ## Start the timing for the matrix and rhs construction##
        if self.pb_formulation == "juffer":
            A, rhs_1, rhs_2 = pb_formulation.juffer(dirichl_space, neumann_space, self.q, self.x_q, self.ep_in, self.ep_ex, self.kappa)
        elif self.pb_formulation == "direct":
            A, rhs_1, rhs_2 = pb_formulation.direct(dirichl_space, neumann_space, self.q, self.x_q, self.ep_in, self.ep_ex, self.kappa)
        elif self.pb_formulation == "alpha_beta":
            A, rhs_1, rhs_2, A_in, A_ex, interior_projector, scaled_exterior_projector = pb_formulation.alpha_beta(dirichl_space, neumann_space, self.q, self.x_q, self.ep_in, self.ep_ex, self.kappa, self.pb_formulation_alpha, self.pb_formulation_beta)
        self.time_matrix_and_rhs_construction = time.time()-setup_start_time
        
        
        ## Check to see if preconditioning is to be applied ##
        preconditioning_start_time = time.time()
        if self.pb_formulation_preconditioning and self.pb_formulation == "alpha_beta":
            if self.pb_formulation_preconditioning_type == "interior":
                A_conditioner = A_in
            elif self.pb_formulation_preconditioning_type == "exterior":
                A_conditioner = A_ex
            elif self.pb_formulation_preconditioning_type == "scaled_exterior_projector":
                A_conditioner = scaled_exterior_projector
            elif self.pb_formulation_preconditioning_type == "interior_projector":
                A_conditioner = interior_projector
            elif self.pb_formulation_preconditioning_type == "squared":
                A_conditioner = A
            else:
                raise ValueError('Unrecognised preconditioning type')

            A_final = A_conditioner * A
            rhs = A_conditioner * [rhs_1, rhs_2]

        ## Set variables for system of equations if no preconditioning is to applied ##
        else:
            A_final = A
            rhs = [rhs_1, rhs_2]
        self.time_preconditioning = time.time()-preconditioning_start_time
        
        
        ## Pass matrix A to discrete form (either strong or weak) ##
        matrix_discrete_start_time = time.time()
        A_discrete = matrix_to_discrete_form(A_final, self.discrete_form_type)
        rhs_discrete = rhs_to_discrete_form(rhs, self.discrete_form_type, A)
        self.time_matrix_to_discrete = time.time()-matrix_discrete_start_time
        
        
        
        ## Use GMRES to solve the system of equations ##
        gmres_start_time = time.time()
        x, info, it_count = utils.solver(A_discrete, rhs_discrete, self.gmres_tolerance, self.gmres_max_iterations)
        self.time_gmres = time.time()-gmres_start_time
        
        ## Split solution and generate corresponding grid functions
        from bempp.api.assembly.blocked_operator import grid_function_list_from_coefficients
        (dirichlet_solution, neumann_solution) = grid_function_list_from_coefficients(x.ravel(), A.domain_spaces)
        
        ## Save number of iterations taken and the solution of the system ##
        self.solver_iteration_count = it_count
        self.phi = dirichlet_solution
        self.d_phi = neumann_solution
        
        ## Finished computing surface potential, register total time taken ##
        self.time_compue_potential = time.time()-start_time
        
        ## Print times, if this is desiered ##
        if self.print_times:
            print('It took ', self.time_matrix_and_rhs_construction, ' seconds to construct the matrices and rhs vectores')
            print('It took ', self.time_matrix_to_discrete, ' seconds to pass the main matrix to discrete form ('+self.discrete_form_type+')')
            print('It took ', self.time_preconditioning, ' seconds to compute and apply the preconditioning ('+str(self.pb_formulation_preconditioning)+')('+self.pb_formulation_preconditioning_type+')')
            print('It took ', self.time_gmres, ' seconds to resolve the system using GMRES')
            print('It took ', self.time_compue_potential, ' seconds in total to compute the surface potential')


            
    def calculate_solvation_energy(self):
        if not hasattr(self, 'phi'):
            ## If surface potential has not been calculated, calculate it now ##
            self.calculate_potential()
            
        start_time = time.time()
        dirichl_space = self.dirichl_space
        neumann_space = self.neumann_space
        
        solution_dirichl = self.phi
        solution_neumann = self.d_phi

        from bempp.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(neumann_space, self.x_q.transpose())
        dlp_q = double_layer(dirichl_space, self.x_q.transpose())
        phi_q = slp_q*solution_neumann - dlp_q*solution_dirichl

        # total solvation energy applying constant to get units [kcal/mol]
        total_energy = 2*np.pi*332.064*np.sum(self.q*phi_q).real

        self.solvation_energy = total_energy
        
        self.time_calc_energy = time.time()-start_time
        if self.print_times:
            print('It took ', self.time_calc_energy, ' seconds to compute the solvatation energy')

            

def generate_msms_mesh_import_charges(solute):
    mesh_dir = os.path.abspath("mesh_temp/")
    if solute.save_mesh_build_files:
        mesh_dir = solute.mesh_build_files_dir

    if not os.path.exists(mesh_dir):
        try:
            os.mkdir(mesh_dir)
        except OSError:
            print ("Creation of the directory %s failed" % mesh_dir)

    if solute.imported_file_type == "pdb":
        mesh_pqr_path = os.path.join(mesh_dir, solute.solute_name+".pqr")
        mesh_tools.convert_pdb2pqr(solute.pdb_path, mesh_pqr_path, solute.force_field)
    else:
        mesh_pqr_path = solute.pqr_path

    mesh_xyzr_path = os.path.join(mesh_dir, solute.solute_name+".xyzr")
    mesh_tools.convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path)

    mesh_face_path = os.path.join(mesh_dir, solute.solute_name+".face")
    mesh_vert_path = os.path.join(mesh_dir, solute.solute_name+".vert")
    
    if solute.mesh_generator == "msms":
        mesh_tools.generate_msms_mesh(mesh_xyzr_path, mesh_dir, solute.solute_name, solute.mesh_density, solute.mesh_probe_radius)
    elif solute.mesh_generator == "nanoshaper":
        mesh_tools.generate_nanoshaper_mesh(mesh_xyzr_path, mesh_dir, solute.solute_name, solute.mesh_density, solute.mesh_probe_radius, solute.save_mesh_build_files)
        
    mesh_off_path = os.path.join(mesh_dir, solute.solute_name+".off")
    mesh_tools.convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path)

    grid = mesh_tools.import_msms_mesh(mesh_face_path, mesh_vert_path)
    q, x_q = utils.import_charges(mesh_pqr_path)

    if solute.save_mesh_build_files:
        if solute.imported_file_type == "pdb":
            solute.mesh_pqr_path = mesh_pqr_path
        solute.mesh_xyzr_path = mesh_xyzr_path
        solute.mesh_face_path = mesh_face_path
        solute.mesh_vert_path = mesh_vert_path
        solute.mesh_off_path = mesh_off_path
    else:
        if solute.imported_file_type == "pdb":
            os.remove(mesh_pqr_path)
        os.remove(mesh_xyzr_path)
        os.remove(mesh_face_path)
        os.remove(mesh_vert_path)
        os.remove(mesh_off_path)
        os.rmdir(mesh_dir)

    return grid, q, x_q


def import_charges(solute):
    mesh_dir = os.path.abspath("mesh_temp/")
    if solute.save_mesh_build_files:
        mesh_dir = solute.mesh_build_files_dir

    if not os.path.exists(mesh_dir):
        try:
            os.mkdir(mesh_dir)
        except OSError:
            print ("Creation of the directory %s failed" % mesh_dir)

    if solute.imported_file_type == "pdb":
        mesh_pqr_path = os.path.join(mesh_dir, solute.solute_name+".pqr")
        mesh_tools.convert_pdb2pqr(solute.pdb_path, mesh_pqr_path, solute.force_field)
    else:
        mesh_pqr_path = solute.pqr_path

    q, x_q = utils.import_charges(mesh_pqr_path)
    
    return q, x_q


def get_name_from_pdb(pdb_path):
    pdb_file = open(pdb_path)
    firstline = pdb_file.readline()
    firstline_split = re.split(r'\s{2,}', firstline)
    solute_name = firstline_split[3].lower()
    pdb_file.close()

    return solute_name


def matrix_to_discrete_form(matrix, discrete_form_type):
    if discrete_form_type == "strong":
        matrix_discrete = matrix.strong_form()
    elif discrete_form_type == "weak":
        matrix_discrete = matrix.weak_form()
        
    return matrix_discrete


def rhs_to_discrete_form(rhs_list, discrete_form_type, A):
    from bempp.api.assembly.blocked_operator import coefficients_from_grid_functions_list, projections_from_grid_functions_list
    
    if discrete_form_type == "strong":
        rhs = coefficients_from_grid_functions_list(rhs_list)
    elif discrete_form_type == "weak":
        rhs = projections_from_grid_functions_list(rhs_list, A.dual_to_range_spaces)
        
    return rhs