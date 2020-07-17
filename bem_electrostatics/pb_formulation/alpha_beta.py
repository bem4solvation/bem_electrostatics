import numpy as np
import bempp.api

def laplaceMultitrace(dirichl_space, neumann_space, operator_assembler):
    from bempp.api.operators.boundary import laplace

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0)*laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    A[0, 1] = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    A[1, 0] = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    A[1, 1] = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)

    return A

def modHelmMultitrace(dirichl_space, neumann_space, kappa, operator_assembler):
    from bempp.api.operators.boundary import modified_helmholtz

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0)*modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler)
    A[0, 1] = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler)
    A[1, 0] = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa, assembler=operator_assembler)
    A[1, 1] = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa, assembler=operator_assembler)

    return A

def alpha_beta(dirichl_space, neumann_space, q, x_q, ep_in, ep_ex, kappa, alpha, beta, operator_assembler):
    from bempp.api.operators.boundary import sparse
    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex/ep_in

    A_in = laplaceMultitrace(dirichl_space, neumann_space, operator_assembler)
    A_ex = modHelmMultitrace(dirichl_space, neumann_space, kappa, operator_assembler)

    D = bempp.api.BlockedOperator(2, 2)
    D[0, 0] = alpha*phi_id
    D[0, 1] = 0.0*phi_id
    D[1, 0] = 0.0*phi_id
    D[1, 1] = beta*dph_id

    E = bempp.api.BlockedOperator(2, 2)
    E[0, 0] = phi_id
    E[0, 1] = 0.0*phi_id
    E[1, 0] = 0.0*phi_id
    E[1, 1] = dph_id*(1.0/ep)

    F = bempp.api.BlockedOperator(2, 2)
    F[0, 0] = alpha*phi_id
    F[0, 1] = 0.0*phi_id
    F[1, 0] = 0.0*phi_id
    F[1, 1] = dph_id*(beta/ep)

    Id = bempp.api.BlockedOperator(2, 2)
    Id[0, 0] = phi_id
    Id[0, 1] = 0.0*phi_id
    Id[1, 0] = 0.0*phi_id
    Id[1, 1] = dph_id

    interior_projector = ((0.5*Id)+A_in)
    scaled_exterior_projector = (D*((0.5*Id)-A_ex)*E)
    A = ((0.5*Id)+A_in)+(D*((0.5*Id)-A_ex)*E)-(Id+F)
    
    @bempp.api.real_callable
    def d_green_func(x, n, domain_index, result):
        nrm = np.sqrt((x[0]-x_q[:,0])**2 + (x[1]-x_q[:,1])**2 + (x[2]-x_q[:,2])**2)
        
        const = -1./(4.*np.pi*ep_in)
        result[:] = (-1.0)*const*np.sum(q*np.dot(x-x_q, n)/(nrm**3))

    @bempp.api.real_callable
    def green_func(x, n, domain_index, result):
        nrm = np.sqrt((x[0]-x_q[:,0])**2 + (x[1]-x_q[:,1])**2 + (x[2]-x_q[:,2])**2)
        
        result[:] = (-1.0)*np.sum(q/nrm)/(4.*np.pi*ep_in)

    rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
    rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)

    return A, rhs_1, rhs_2, A_in, A_ex, interior_projector, scaled_exterior_projector

def block_diagonal_preconditioner_alpha_beta(dirichl_space, neumann_space, ep_in, ep_ex, kappa, alpha, beta):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import factorized, LinearOperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    hlp_in_diag = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A
    adlp_in_diag = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A


    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
    dlp_out_diag = (-1.0)*modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
    hlp_out_diag = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa, assembler="only_diagonal_part").weak_form().A
    adlp_out_diag = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa, assembler="only_diagonal_part").weak_form().A

    phi_identity_diag = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal()
    dph_identity_diag = sparse.identity(neumann_space, neumann_space, neumann_space).weak_form().A.diagonal()

    ep = ep_ex/ep_in
    
    diag11 = diags((0.5-(0.5/alpha))*phi_identity_diag + (alpha*dlp_out_diag) - dlp_in_diag)
    diag12 = diags(slp_in_diag - ((alpha/ep)*slp_out_diag))
    diag21 = diags(hlp_in_diag - (beta*hlp_out_diag))
    diag22 = diags(-(0.5+(beta/(2*ep)))*dph_identity_diag + hlp_in_diag - ((beta/ep)*hlp_out_diag))
    block_mat_precond = bmat([[diag11, diag12], [diag21, diag22]]).tocsr()  # csr_matrix

    solve = factorized(block_mat_precond)  # a callable for solving a sparse linear system (treat it as an inverse)
    precond = LinearOperator(matvec=solve, dtype='float64', shape=block_mat_precond.shape)
    
    return precond