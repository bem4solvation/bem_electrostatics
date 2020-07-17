import numpy as np
import bempp.api

def direct(dirichl_space, neumann_space, q, x_q, ep_in, ep_out, kappa, operator_assembler): 
    
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler)

    # Matrix Assembly
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*identity + dlp_in
    blocked[0, 1] = -slp_in
    blocked[1, 0] = 0.5*identity - dlp_out
    blocked[1, 1] = (ep_in/ep_out)*slp_out
    
    A = blocked
   
    
    @bempp.api.real_callable
    def charges_fun(x, n, domain_index, result):
        nrm = np.sqrt((x[0]-x_q[:,0])**2 + (x[1]-x_q[:,1])**2 + (x[2]-x_q[:,2])**2)
        aux = np.sum(q/nrm)
        
        result[0] = aux/(4*np.pi*ep_in)
        
    @bempp.api.real_callable
    def zero(x, n, domain_index, result):
        result[0] = 0

    rhs_1 = bempp.api.GridFunction(dirichl_space, fun=charges_fun)
    rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)

    return A, rhs_1, rhs_2

def block_diagonal_preconditioner(dirichl_space, neumann_space, ep_in, ep_ex, kappa):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import factorized, LinearOperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
    
    # block-diagonal preconditioner
    identity_diag = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal()
    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
    dlp_out_diag = modified_helmholtz.double_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A

    diag11 = diags(.5 * identity_diag + dlp_in_diag)
    diag12 = diags(-slp_in_diag)
    diag21 = diags(.5 * identity_diag - dlp_out_diag)
    diag22 = diags((ep_in / ep_ex) * slp_out_diag)
    block_mat_precond = bmat([[diag11, diag12], [diag21, diag22]]).tocsr()  # csr_matrix

    solve = factorized(block_mat_precond)  # a callable for solving a sparse linear system (treat it as an inverse)
    precond = LinearOperator(matvec=solve, dtype='float64', shape=block_mat_precond.shape)
    
    return precond