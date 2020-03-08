import numpy as np
import bempp.api

def direct(dirichl_space, neumann_space, q, x_q, ep_in, ep_out, kappa): 
    
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa)

    # Matrix Assembly
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*identity + dlp_in
    blocked[0, 1] = -slp_in
    blocked[1, 0] = 0.5*identity - dlp_out
    blocked[1, 1] = (ep_in/ep_out)*slp_out
    #A = blocked.strong_form()
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