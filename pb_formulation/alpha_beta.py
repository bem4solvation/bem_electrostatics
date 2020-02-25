import numpy as np
import bempp.api

def laplaceMultitrace(dirichl_space, neumann_space):
    from bempp.api.operators.boundary import laplace

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0)*laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)
    A[0, 1] = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
    A[1, 0] = laplace.hypersingular(dirichl_space, neumann_space, neumann_space)
    A[1, 1] = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space)

    return A

def modHelmMultitrace(dirichl_space, neumann_space, kappa):
    from bempp.api.operators.boundary import modified_helmholtz

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0)*modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa)
    A[0, 1] = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa)
    A[1, 0] = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa)
    A[1, 1] = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa)

    return A

def alpha_beta(dirichl_space, neumann_space, q, x_q, ep_in, ep_out, kappa, alpha, beta):

    from bempp.api.operators.boundary import sparse
    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_out/ep_in

    A_in = laplaceMultitrace(dirichl_space, neumann_space)
    A_out = modHelmMultitrace(dirichl_space, neumann_space, kappa)

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

    A = ((0.5*Id)+A_in)+(D*((0.5*Id)-A_out)*E)-(Id+F)
    #A = A.strong_form()
    print("Got A")
    
    @bempp.api.real_callable
    def d_green_func(x, n, domain_index, result):
        
        F = (x-x_q)
        z = np.zeros(F.shape[0], dtype=np.float64)
        for i in range(F.shape[0]):
            nrm = np.linalg.norm(F[i, :])
            z[i] = nrm
    
        #result[:] = np.sum(q/x)/(4*np.pi*ep_in)
        
        const = -1./(4.*np.pi*ep_in)
        result[:] = (-1.0)*const*np.sum(q*np.dot( x - x_q, n )/(np.linalg.norm( x - x_q, axis=1 )**3))
        #result[:] = (-1.0)*const*np.sum(q*np.dot(x-x_q, n)/(z**3))

    @bempp.api.real_callable
    def green_func(x, n, domain_index, result):
        F = (x-x_q)
        z = np.zeros(F.shape[0], dtype=np.float64)
        for i in range(F.shape[0]):
            nrm = np.linalg.norm(F[i, :])
            z[i] = nrm
            
        result[:] = (-1.0)*np.sum(q/np.linalg.norm( x - x_q, axis=1 ))/(4.*np.pi*ep_in)
        #result[:] = (-1.0)*np.sum(q/z)/(4.*np.pi*ep_in)

    print("start RHS1")
    rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
    print("start RHS2")
    rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)
    #rhs = np.concatenate([rhs_1.coefficients, rhs_2.coefficients])
    print("RHS done")

    return A, rhs_1, rhs_2