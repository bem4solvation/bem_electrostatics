import numpy as np
import bempp.api

def juffer(dirichl_space, neumann_space, q, x_q, ep_in, ep_ex, kappa):
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)
    ep = ep_ex/ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa)
    L1 = ep*dP - dF

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa)
    L2 = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space)
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa)
    L3 = ddP - ddF

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space)
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa)
    L4 = dF0 - (1./ep)*dP0

    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*(1. + ep)*phi_id - L1
    blocked[0, 1] = -L2
    blocked[1, 0] = L3    # Cambio de signo por definicion de bempp
    blocked[1, 1] = 0.5*(1. + 1./ep)*dph_id - L4
    #A = blocked.strong_form()
    A = blocked

    def d_green_func(x, n, domain_index, result):
        const = -1./(4.*np.pi*ep_in)
        result[:] = const*np.sum(q*np.dot( x - x_q, n )/(np.linalg.norm( x - x_q, axis=1 )**3))

    def green_func(x, n, domain_index, result):
        result[:] = np.sum(q/np.linalg.norm( x - x_q, axis=1 ))/(4.*np.pi*ep_in)

    rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
    rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)
    rhs = np.concatenate([rhs_1.coefficients, rhs_2.coefficients])

    return A, rhs