import numpy as np


def calderon(A, interior_op, exterior_op, interior_projector, scaled_exterior_projector, formulation,
             preconditioning_type):
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
        raise ValueError('Calderon preconditioning only implemented for alpha_beta formulation')
        
    return A_conditioner


def first_kind(A, preconditioning_type, dirichl_space, neumann_space, ep_in, ep_ex, kappa, operator_assembler):
    import bem_electrostatics.pb_formulation.formulations.lhs as lhs
    if preconditioning_type == "calderon_squared":
        A_conditioner = A
    elif preconditioning_type == "calderon_interior_operator":
        A_conditioner = lhs.laplace_multitrace(dirichl_space, neumann_space, operator_assembler)
    elif preconditioning_type == "calderon_exterior_operator":
        A_conditioner = lhs.mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler)
    elif preconditioning_type == "calderon_interior_operator_scaled":
        scaling_factors = [[1.0, (ep_ex/ep_in)],
                           [(ep_in/ep_ex), 1.0]]
        A_conditioner = lhs.laplace_multitrace_scaled(dirichl_space, neumann_space, scaling_factors, operator_assembler)
    elif preconditioning_type == "calderon_exterior_operator_scaled":
        scaling_factors = [[1.0, (ep_in/ep_ex)],
                           [(ep_ex/ep_in), 1.0]]
        A_conditioner = lhs.mod_helm_multitrace_scaled(dirichl_space, neumann_space, kappa, scaling_factors, operator_assembler)
    else:
        raise ValueError('Calderon preconditioning type not recognised.')

    return A_conditioner


def calderon_scaled_mass(preconditioning_type, formulation_type, dirichl_space, neumann_space, ep_in, ep_ex, kappa,
                         operator_assembler):
    import bem_electrostatics.pb_formulation.formulations.lhs as lhs

    def mass_matrix():
        if formulation_type.startswith("first_kind_internal"):
            mass = first_kind_interior_scaled_mass(dirichl_space, ep_in, ep_ex, preconditioner)
        elif formulation_type.startswith("first_kind_external"):
            mass = first_kind_exterior_scaled_mass(dirichl_space, ep_in, ep_ex, preconditioner)
        else:
            raise ValueError('Calderon preconditioning type not recognised.')

        return mass

    if preconditioning_type == "calderon_scaled_interior_operator":
        preconditioner = lhs.laplace_multitrace(dirichl_space, neumann_space, operator_assembler)
        preconditioner_with_mass = mass_matrix() * preconditioner.weak_form()
    elif preconditioning_type == "calderon_scaled_exterior_operator":
        preconditioner = lhs.mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler)
        preconditioner_with_mass = mass_matrix() * preconditioner.weak_form()
    elif preconditioning_type == "calderon_scaled_interior_operator_scaled":
        scaling_factors = [[1.0, (ep_ex / ep_in)], [(ep_in / ep_ex), 1.0]]
        preconditioner = lhs.laplace_multitrace_scaled(dirichl_space, neumann_space, scaling_factors,
                                                       operator_assembler)
        preconditioner_with_mass = mass_matrix() * preconditioner.weak_form()
    elif preconditioning_type == "calderon_scaled_exterior_operator_scaled":
        scaling_factors = [[1.0, (ep_in / ep_ex)], [(ep_ex / ep_in), 1.0]]
        preconditioner = lhs.mod_helm_multitrace_scaled(dirichl_space, neumann_space, kappa, scaling_factors,
                                                        operator_assembler)
        preconditioner_with_mass = mass_matrix() * preconditioner.weak_form()
    else:
        raise ValueError('Calderon preconditioning type not recognised.')

    return preconditioner_with_mass


def block_diagonal(dirichl_space, neumann_space, ep_in, ep_ex, kappa, formulation_type, alpha, beta):
    if formulation_type == "direct":
        preconditioner = block_diagonal_precon_direct(dirichl_space, neumann_space, ep_in, ep_ex, kappa)
    elif formulation_type == "direct_permuted":
        preconditioner = block_diagonal_precon_direct(dirichl_space, neumann_space, ep_in, ep_ex, kappa, True)
    elif formulation_type == "direct_external":
        preconditioner = block_diagonal_precon_direct_external(dirichl_space, neumann_space, ep_in, ep_ex, kappa)
    elif formulation_type == "direct_external_permuted":
        preconditioner = block_diagonal_precon_direct_external(dirichl_space, neumann_space, ep_in, ep_ex, kappa, True)
    elif formulation_type == "juffer":
        preconditioner = block_diagonal_precon_juffer(dirichl_space, neumann_space, ep_in, ep_ex, kappa)
    elif formulation_type == "alpha_beta":
        preconditioner = block_diagonal_precon_alpha_beta(dirichl_space, neumann_space, ep_in, ep_ex, kappa, alpha,
                                                          beta)
    else:
        raise ValueError('Block-diagonal preconditioning not implemented for the given formulation type.')
    
    return preconditioner


def juffer_scaled_mass(dirichl_space, ep_in, ep_ex, matrix):
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    nrows = len(matrix.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(matrix.range_spaces[index],
                                                          matrix.dual_to_range_spaces[index])

    range_ops[0, 0] = range_ops[0, 0] * (1.0 / (0.5 * (1.0 + (ep_ex/ep_in))))
    range_ops[1, 1] = range_ops[1, 1] * (1.0 / (0.5*(1.0+(ep_in/ep_ex))))

    preconditioner = BlockedDiscreteOperator(range_ops)

    return preconditioner


def first_kind_interior_scaled_mass(dirichl_space, ep_in, ep_ex, matrix):
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    nrows = len(matrix.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(matrix.range_spaces[index],
                                                          matrix.dual_to_range_spaces[index])

    range_ops[0, 0] = range_ops[0, 0] * (1.0 / (0.25 + (ep_ex/(4.0*ep_in))))
    range_ops[1, 1] = range_ops[1, 1] * (1.0 / (0.25+(ep_in/(4.0*ep_ex))))

    preconditioner = BlockedDiscreteOperator(range_ops)

    return preconditioner


def first_kind_exterior_scaled_mass(dirichl_space, ep_in, ep_ex, matrix):
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    nrows = len(matrix.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(matrix.range_spaces[index],
                                                          matrix.dual_to_range_spaces[index])

    range_ops[0, 0] = range_ops[0, 0] * (1.0 / (0.25+(ep_in/(4.0*ep_ex))))
    range_ops[1, 1] = range_ops[1, 1] * (1.0 / (0.25 + (ep_ex/(4.0*ep_in))))

    preconditioner = BlockedDiscreteOperator(range_ops)

    return preconditioner


def block_diagonal_precon_direct_test(solute):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import aslinearoperator

    block1 = solute.matrices['A'][0, 0]
    block2 = solute.matrices['A'][0, 1]
    block3 = solute.matrices['A'][1, 0]
    block4 = solute.matrices['A'][1, 1]

    diag11 = block1._op1._alpha * block1._op1._op.weak_form().to_sparse().diagonal() + \
             block1._op2.descriptor.singular_part.weak_form().to_sparse().diagonal()
    diag12 = block2._alpha * block2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    diag21 = block3._op1._alpha * block3._op1._op.weak_form().to_sparse().diagonal() +\
             block3._op2._alpha * block3._op2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    diag22 = block4._alpha * block4._op.descriptor.singular_part.weak_form().to_sparse().diagonal()

    d_aux = 1 / (diag22 - diag21 * diag12 / diag11)
    diag11_inv = 1 / diag11 + 1 / diag11 * diag12 * d_aux * diag21 / diag11
    diag12_inv = -1 / diag11 * diag12 * d_aux
    diag21_inv = -d_aux * diag21 / diag11
    diag22_inv = d_aux

    block_mat_precond = bmat([[diags(diag11_inv), diags(diag12_inv)], [diags(diag21_inv), diags(diag22_inv)]]).tocsr()

    return aslinearoperator(block_mat_precond)


def block_diagonal_precon_direct(dirichl_space, neumann_space, ep_in, ep_ex, kappa, permuted_rows = False):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import aslinearoperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

    # block-diagonal preconditioner
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    identity_diag = identity.weak_form().to_sparse().diagonal()
    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_out_diag = modified_helmholtz.double_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()

    if permuted_rows:
        diag11 = .5 * identity_diag - dlp_out_diag
        diag12 = (ep_in / ep_ex) * slp_out_diag
        diag21 = .5 * identity_diag + dlp_in_diag
        diag22 = -slp_in_diag
    else:
        diag11 = .5 * identity_diag + dlp_in_diag
        diag12 = -slp_in_diag
        diag21 = .5 * identity_diag - dlp_out_diag
        diag22 = (ep_in / ep_ex) * slp_out_diag

    d_aux = 1 / (diag22 - diag21 * diag12 / diag11)
    diag11_inv = 1 / diag11 + 1 / diag11 * diag12 * d_aux * diag21 / diag11
    diag12_inv = -1 / diag11 * diag12 * d_aux
    diag21_inv = -d_aux * diag21 / diag11
    diag22_inv = d_aux

    block_mat_precond = bmat([[diags(diag11_inv), diags(diag12_inv)], [diags(diag21_inv), diags(diag22_inv)]]).tocsr()

    return aslinearoperator(block_mat_precond)


def block_diagonal_precon_direct_external(dirichl_space, neumann_space, ep_in, ep_ex, kappa, permuted_rows = False):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import aslinearoperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

    # block-diagonal preconditioner
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    identity_diag = identity.weak_form().to_sparse().diagonal()
    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_out_diag = modified_helmholtz.double_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()

    if permuted_rows:
        diag11 = 0.5 * identity_diag + dlp_in_diag
        diag12 = -(ep_ex / ep_in) * slp_in_diag
        diag21 = 0.5 * identity_diag - dlp_out_diag
        diag22 = slp_out_diag
    else:
        diag11 = 0.5 * identity_diag - dlp_out_diag
        diag12 = slp_out_diag
        diag21 = 0.5 * identity_diag + dlp_in_diag
        diag22 = -(ep_ex / ep_in) * slp_in_diag

    d_aux = 1 / (diag22 - diag21 * diag12 / diag11)
    diag11_inv = 1 / diag11 + 1 / diag11 * diag12 * d_aux * diag21 / diag11
    diag12_inv = -1 / diag11 * diag12 * d_aux
    diag21_inv = -d_aux * diag21 / diag11
    diag22_inv = d_aux

    block_mat_precond = bmat([[diags(diag11_inv), diags(diag12_inv)],
                              [diags(diag21_inv), diags(diag22_inv)]]).tocsr()

    return aslinearoperator(block_mat_precond)


def block_diagonal_precon_juffer(dirichl_space, neumann_space, ep_in, ep_ex, kappa):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import factorized, LinearOperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal()
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space).weak_form().A.diagonal()
    ep = ep_ex/ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                              assembler="only_diagonal_part").weak_form().A
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                         assembler="only_diagonal_part").weak_form().A
    L1 = (ep*dP) - dF

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                             assembler="only_diagonal_part").weak_form().A
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                        assembler="only_diagonal_part").weak_form().A
    L2 = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space,
                                assembler="only_diagonal_part").weak_form().A
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                           assembler="only_diagonal_part").weak_form().A
    L3 = ddP - ddF

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space,
                                       assembler="only_diagonal_part").weak_form().A
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                  assembler="only_diagonal_part").weak_form().A
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
    
    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().A
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().A
    hlp_in_diag = laplace.hypersingular(dirichl_space, neumann_space, neumann_space,
                                        assembler="only_diagonal_part").weak_form().A
    adlp_in_diag = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space,
                                                assembler="only_diagonal_part").weak_form().A
    
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().A
    dlp_out_diag = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().A
    hlp_out_diag = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                                    assembler="only_diagonal_part").weak_form().A
    adlp_out_diag = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                            assembler="only_diagonal_part").weak_form().A

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
