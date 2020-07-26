def calderon(A, interior_op, exterior_op, interior_projector, scaled_exterior_projector, formulation, preconditioning_type):
    if formulation == "alpha_beta":
        if preconditioning_type == "calderon_squared":
            A_conditioner = A
        elif preconditioning_type == "calderon_interior_operator":
            A_conditioner = interior_op
        elif preconditioning_type == "calderon_exterior_operator":
            A_conditioner = exterior_op
        elif preconditioning_type == "calderon_interior_projector":
            A_conditioner = interior_projector
        elif preconditioning_type == "calderon_scaled_exterior_projector":
            A_conditioner = scaled_exterior_projector
        else:
            raise ValueError('Calderon preconditioning type not recognised.')
    else:
        raise ValueError('Calderon precondionting only implemented for alpha_beta formulation')
        
    return A_conditioner

def block_diagonal(dirichl_space, neumann_space, ep_in, ep_ex, kappa, formulation_type, alpha, beta):
    if formulation_type == "direct":
        preconditioner = block_diagonal_precon_direct(dirichl_space, neumann_space, ep_in, ep_ex, kappa)
    elif formulation_type == "juffer":
        preconditioner = block_diagonal_precon_juffer(dirichl_space, neumann_space, ep_in, ep_ex, kappa)
    elif formulation_type == "alpha_beta":
        preconditioner = block_diagonal_precon_alpha_beta(dirichl_space, neumann_space, ep_in, ep_ex, kappa, alpha, beta)
    else:
        raise ValueError('Block-diagonal preconditioning not implemented for the given formulation type.')
    
    return preconditioner

def block_diagonal_precon_direct(dirichl_space, neumann_space, ep_in, ep_ex, kappa):
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

def block_diagonal_precon_juffer(dirichl_space, neumann_space, ep_in, ep_ex, kappa):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import factorized, LinearOperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal()
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space).weak_form().A.diagonal()
    ep = ep_ex/ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
    L1 = (ep*dP) - dF

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
    L2 = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa, assembler="only_diagonal_part").weak_form().A
    L3 = ddP - ddF

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa, assembler="only_diagonal_part").weak_form().A
    L4 = dF0 - ((1.0/ep)*dP0)

    diag11 = diags((0.5*(1.0 + ep)*phi_id) - L1)
    diag12 = diags((-1.0)*L2)
    diag21 = diags(L3)
    diag22 = diags((0.5*(1.0 + (1.0/ep))*dph_id) - L4)
    block_mat_precond = bmat([[diag11, diag12], [diag21, diag22]]).tocsr()  # csr_matrix

    solve = factorized(block_mat_precond)  # a callable for solving a sparse linear system (treat it as an inverse)
    precond = LinearOperator(matvec=solve, dtype='float64', shape=block_mat_precond.shape)
    
    return precond

def block_diagonal_precon_alpha_beta(dirichl_space, neumann_space, ep_in, ep_ex, kappa, alpha, beta):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import factorized, LinearOperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
    
    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
    hlp_in_diag = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A
    adlp_in_diag = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler="only_diagonal_part").weak_form().A
    
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
    dlp_out_diag = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
    hlp_out_diag = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa, assembler="only_diagonal_part").weak_form().A
    adlp_out_diag = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa, assembler="only_diagonal_part").weak_form().A

    phi_identity_diag = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal()
    dph_identity_diag = sparse.identity(neumann_space, neumann_space, neumann_space).weak_form().A.diagonal()

    ep = ep_ex/ep_in
    
    diag11 = diags((-0.5*(1+alpha))*phi_identity_diag + (alpha*dlp_out_diag) - dlp_in_diag)
    diag12 = diags(slp_in_diag - ((alpha/ep)*slp_out_diag))
    diag21 = diags(hlp_in_diag - (beta*hlp_out_diag))
    diag22 = diags((-0.5*(1+(beta/ep)))*dph_identity_diag + adlp_in_diag - ((beta/ep)*adlp_out_diag))
    block_mat_precond = bmat([[diag11, diag12], [diag21, diag22]]).tocsr()  # csr_matrix

    solve = factorized(block_mat_precond)  # a callable for solving a sparse linear system (treat it as an inverse)
    precond = LinearOperator(matvec=solve, dtype='float64', shape=block_mat_precond.shape)
    
    return precond

