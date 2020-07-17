import numpy as np

def import_charges(pqr_path):
    # Read charges and coordinates from the .pqr file
    q, x_q = np.array([]), np.empty((0,3))
    molecule_file = open(pqr_path, 'r')
    molecule_data = molecule_file.read().split('\n')
    for line in molecule_data:
        line = line.split()
        if len(line)==0 or line[0]!='ATOM': continue
        q = np.append( q, float(line[8]))
        x_q = np.vstack((x_q, np.array(line[5:8]).astype(float)))

    return q, x_q

def solver(A, rhs, tolerance, max_iterations, precond=None):
    from scipy.sparse.linalg import gmres
    from bempp.api.linalg.iterative_solvers import IterationCounter
    
    callback = IterationCounter(True)
    
    if precond == None:
        x, info = gmres(A, rhs, tol=tolerance, maxiter=max_iterations, callback=callback)
    else:
        x, info = gmres(A, rhs, M=precond, tol=tolerance, maxiter=max_iterations, callback=callback)

    return x, info, callback.count