import numpy as np
import bempp.api

from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz


def direct(dirichl_space, neumann_space, ep_in, ep_out, kappa, operator_assembler):
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    dlp_in = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    slp_out = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)
    dlp_out = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = 0.5 * identity + dlp_in
    A[0, 1] = -slp_in
    A[1, 0] = 0.5 * identity - dlp_out
    A[1, 1] = (ep_in / ep_out) * slp_out

    return A


def juffer(dirichl_space, neumann_space, ep_in, ep_ex, kappa, operator_assembler):
    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)
    ep = ep_ex / ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                         assembler=operator_assembler)
    L1 = (ep * dP) - dF

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                        assembler=operator_assembler)
    L2 = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                           assembler=operator_assembler)
    L3 = ddP - ddF

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                  assembler=operator_assembler)
    L4 = dF0 - ((1.0 / ep) * dP0)

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (0.5 * (1.0 + ep) * phi_id) - L1
    A[0, 1] = (-1.0) * L2
    A[1, 0] = L3  # Sign change due to bempp definition
    A[1, 1] = (0.5 * (1.0 + (1.0 / ep)) * dph_id) - L4

    return A


def laplace_multitrace(dirichl_space, neumann_space, operator_assembler):
    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0) * laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    A[0, 1] = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    A[1, 0] = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    A[1, 1] = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)

    return A


def mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler):
    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0) * modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                                       assembler=operator_assembler)
    A[0, 1] = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)
    A[1, 0] = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                               assembler=operator_assembler)
    A[1, 1] = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                      assembler=operator_assembler)

    return A


def alpha_beta(dirichl_space, neumann_space, ep_in, ep_ex, kappa, alpha, beta, operator_assembler):
    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex / ep_in

    A_in = laplace_multitrace(dirichl_space, neumann_space, operator_assembler)
    A_ex = mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler)

    D = bempp.api.BlockedOperator(2, 2)
    D[0, 0] = alpha * phi_id
    D[0, 1] = 0.0 * phi_id
    D[1, 0] = 0.0 * phi_id
    D[1, 1] = beta * dph_id

    E = bempp.api.BlockedOperator(2, 2)
    E[0, 0] = phi_id
    E[0, 1] = 0.0 * phi_id
    E[1, 0] = 0.0 * phi_id
    E[1, 1] = dph_id * (1.0 / ep)

    F = bempp.api.BlockedOperator(2, 2)
    F[0, 0] = alpha * phi_id
    F[0, 1] = 0.0 * phi_id
    F[1, 0] = 0.0 * phi_id
    F[1, 1] = dph_id * (beta / ep)

    Id = bempp.api.BlockedOperator(2, 2)
    Id[0, 0] = phi_id
    Id[0, 1] = 0.0 * phi_id
    Id[1, 0] = 0.0 * phi_id
    Id[1, 1] = dph_id

    interior_projector = ((0.5 * Id) + A_in)
    scaled_exterior_projector = (D * ((0.5 * Id) - A_ex) * E)
    A = ((0.5 * Id) + A_in) + (D * ((0.5 * Id) - A_ex) * E) - (Id + F)

    return A, A_in, A_ex, interior_projector, scaled_exterior_projector


def alpha_beta_external(dirichl_space, neumann_space, ep_in, ep_ex, kappa, alpha, beta, operator_assembler):
    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex / ep_in

    A_in = laplace_multitrace(dirichl_space, neumann_space, operator_assembler)
    A_ex = mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler)

    D = bempp.api.BlockedOperator(2, 2)
    D[0, 0] = alpha * phi_id
    D[0, 1] = 0.0 * phi_id
    D[1, 0] = 0.0 * phi_id
    D[1, 1] = beta * dph_id

    E_1 = bempp.api.BlockedOperator(2, 2)
    E_1[0, 0] = phi_id
    E_1[0, 1] = 0.0 * phi_id
    E_1[1, 0] = 0.0 * phi_id
    E_1[1, 1] = dph_id * ep

    F = bempp.api.BlockedOperator(2, 2)
    F[0, 0] = alpha * phi_id
    F[0, 1] = 0.0 * phi_id
    F[1, 0] = 0.0 * phi_id
    F[1, 1] = dph_id * (beta / ep)

    Id = bempp.api.BlockedOperator(2, 2)
    Id[0, 0] = phi_id
    Id[0, 1] = 0.0 * phi_id
    Id[1, 0] = 0.0 * phi_id
    Id[1, 1] = dph_id

    interior_projector = ((0.5 * Id) + A_in) * E_1
    scaled_exterior_projector = D * ((0.5 * Id) - A_ex)
    A = (((0.5 * Id) + A_in) * E_1) + (D * ((0.5 * Id) - A_ex)) - D - E_1

    return A, A_in, A_ex, interior_projector, scaled_exterior_projector


def alpha_beta_single_blocked_operator(dirichl_space, neumann_space, ep_in, ep_ex, kappa, alpha, beta,
                                       operator_assembler):
    dlp_in = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    slp_in = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    hlp_in = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    adlp_in = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)

    dlp_out = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)
    slp_out = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)
    hlp_out = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                               assembler=operator_assembler)
    adlp_out = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                       assembler=operator_assembler)

    phi_identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_identity = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex / ep_in

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-0.5 * (1 + alpha)) * phi_identity + (alpha * dlp_out) - dlp_in
    A[0, 1] = slp_in - ((alpha / ep) * slp_out)
    A[1, 0] = hlp_in - (beta * hlp_out)
    A[1, 1] = (-0.5 * (1 + (beta / ep))) * dph_identity + adlp_in - ((beta / ep) * adlp_out)

    return A

def derivative_ex(dirichl_space, neumann_space, ep_in, ep_ex, kappa, operator_assembler):
    """
    Construct the system matrix and RHS grid functions using derivative formulation with interior values.
    """
    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)
    ep = ep_ex/ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler)
    B = 1/ep * dF - dP

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler)
    A = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa, assembler=operator_assembler)
    D = 1/ep * (ddP - ddF)

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa, assembler=operator_assembler)
    C = dF0 - 1.0/ep*dP0

    A_sys = bempp.api.BlockedOperator(2, 2)
    A_sys[0, 0] = (0.5*(1.0 + (1.0/ep))*phi_id) + B
    A_sys[0, 1] = -A
    A_sys[1, 0] = D
    A_sys[1, 1] = (0.5*(1.0 + (1.0/ep))*dph_id) - C

    return A_sys
