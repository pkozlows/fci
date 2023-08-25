import time
import numpy as np
from main import generation, integrals


def Davidson(matrix, n_eigval, tolerance = 1e-8):
    # """Takes a sparse, diagonally dominant matrix, like the FCI hamiltonian and the number of desired eigenvalues."""
    # check that the matrix is symmetric
    assert(np.allclose(matrix, matrix.T))
    rows, cols = matrix.shape
    # make our initial guess the hf reference eigenvector
    guess_space = np.eye(rows, n_eigval)
    for i in range(rows // 2):
        # make our gas space or the normal
        guess_space, R = np.linalg.qr(guess_space)
        transformed_space = matrix @ guess_space
        # calculate the rayleigh matrix
        rayleigh_matrix = guess_space.T @ transformed_space
        # calculate the eigenvalues and eigenvectors of the rayleigh matrix
        eigenvalues, eigenvectors = np.linalg.eig(rayleigh_matrix)
        # sort the eigenvalues and eigenvectors
        sorted_indices = eigenvalues.argsort()
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # Compute the matrix products only once for all eigenvectors
        matrix_prods = transformed_space @ eigenvectors

        # Prepare an array for residuals and corrections
        residuals = np.zeros((rows, n_eigval))
        new_directions = np.zeros((rows, n_eigval))

        # Using the diagonal directly
        matrix_diag = np.diag(matrix)

        for j in range(n_eigval):
            # Calculate residuals for the j-th eigenvector
            residuals[:, j] = matrix_prods[:, j] - eigenvalues[j] * guess_space @ eigenvectors[:, j]
        
            # Using the diagonal directly for corrections
            diff = eigenvalues[j] - matrix_diag
            ind = np.abs(diff) >= tolerance
            new_directions[ind, j] = residuals[ind, j] / diff[ind]

        # Check for convergence
        if np.max(np.linalg.norm(residuals, axis=0)) < tolerance:
            return eigenvalues[:n_eigval]

        # Expand the guess space with the new directions
        guess_space = np.hstack((guess_space, new_directions))
    return 
# generate the hamiltonian
started_made_cukes = time.time()
hamiltonian = generation(integrals)
and_met_chokes = time.time()
# fun the time date Davidson da economization takes
start_davidson = time.time()
davison_fetus = Davidson(hamiltonian, 2)
# find the time that Davidson digestion takes
end_davidson = time.time()
# find the time that numpy digestion takes
start_numpy = time.time()
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)# thinner
# find the time that numpy digestion takes
end_numpy = time.time()
# print the times
print("Time to make cukes: ", and_met_chokes - started_made_cukes)
print("Time to Davidson: ", end_davidson - start_davidson)
print("Time to numpy: ", end_numpy - start_numpy)
# print out the results of the Davidson and the numpy
print("Davidson eigenvalues: ", davison_fetus)
print("Numpy eigenvalues: ", eigenvalues[:2])
assert((end_numpy - start_numpy) > (end_davidson - start_davidson))
