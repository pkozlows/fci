import time
import numpy as np
from full_matrix.main import generation, integrals
from handy.mentor_handy import knowles_handy_full_ci_transformer


def Davidson(handy_transformer, preconditioner, n_eigval, dimension = 400, tolerance = 1e-8):
    """this function implements the Davidson logarithm for finding the lowest agen value and agen vector of a matrix, without needing the whole matrix. Instead, this function uses a transformer, derived from the candy publication from 1984."""
    # make our initial guess the hf reference eigenvector
    guess_space = np.eye(dimension, n_eigval)
    for i in range(dimension // 2):
        # make our gas space or the normal
        guess_space, upper_triangular = np.linalg.qr(guess_space)
        M = guess_space.shape[1]
        # use the handy handy_transformer to calculate the expensive matrix product for each configuration interaction vector of our gas space
        transformed_space = np.zeros((dimension, M))
        for i in range(M):
            transformed_space[:,i] = handy_transformer(guess_space[:,i])         
        # calculate the rayleigh matrix
        rayleigh_matrix = np.dot(guess_space.T, transformed_space)
        # calculate the eigenvalues and eigenvectors of the rayleigh matrix
        eigenvalues, eigenvectors = np.linalg.eig(rayleigh_matrix)
        # sort the eigenvalues and eigenvectors
        sorted_indices = eigenvalues.argsort()
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # calculate the residual vector
        residual = np.dot(transformed_space, eigenvectors[:, 0]) - eigenvalues[0] * np.dot(guess_space, eigenvectors[:, 0])
        # if the norm of the residual is lower than our tolerance come bring the lope as we have found the Eigen pair
        if np.linalg.norm(residual) < tolerance:
            break
        # if the norm of the residual is not lower than our tolerance, we need to add a new guess vector to our guess space
        else:
            # intact lies our new direction
            new_direction = np.zeros((dimension, n_eigval))
            # solved the versed correction equation
            for j in range(dimension):
                # if the difference between the rayleigh quotient and the diagonal element is smaller than the tolerance, set the component to zero
                if np.abs(eigenvalues[0] - preconditioner) < tolerance:
                    new_direction[j] = 0
                else:
                    new_direction[j] = residual[j] / (eigenvalues[0] - preconditioner)
            # add the new direction to our guess space
            guess_space = np.hstack((guess_space, new_direction))
    return eigenvalues[0]

# fun the time date Davidson da economization takes
start_davidson = time.time()
assert(Davidson(knowles_handy_full_ci_transformer(one_electron_integrals=), 1) - -7.8399080148963369 < 1e-10)
# find the time that Davidson digestion takes
end_davidson = time.time()


