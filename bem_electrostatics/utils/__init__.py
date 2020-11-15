import numpy as np


def import_charges(pqr_path):
    # Read charges and coordinates from the .pqr file
    molecule_file = open(pqr_path, 'r')
    molecule_data = molecule_file.read().split('\n')
    atom_count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != 'ATOM':
            continue
        atom_count += 1

    q, x_q = np.empty((atom_count, 1)), np.empty((atom_count, 3))
    count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != 'ATOM':
            continue
        q[count] = float(line[8])
        x_q[count, :] = line[5:8]
        count += 1

    return q, x_q


def solver(A, rhs, tolerance, max_iterations, precond=None):
    from scipy.sparse.linalg import gmres
    from bempp.api.linalg.iterative_solvers import IterationCounter

    callback = IterationCounter(True)

    if precond is None:
        x, info = gmres(A, rhs, tol=tolerance, maxiter=max_iterations, callback=callback)
    else:
        x, info = gmres(A, rhs, M=precond, tol=tolerance, maxiter=max_iterations, callback=callback)

    return x, info, callback.count
