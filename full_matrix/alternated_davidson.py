import time
import numpy as np
import numpy as np
from main import generation, integrals


def Davidson(matrix, n_eigval, tolerance = 1e-8):
    """Takes a sparse, diagonally dominant matrix, like the FCI hamiltonian and the number of desired eigenvalues."""
    # check that the matrix is symmetric
    assert(np.allclose(matrix, matrix.T))
    rows, cols = matrix.shape
    # make our initial guess the hf reference eigenvector
    guess_space = np.eye(rows, n_eigval)
    for i in range(rows // 2):
        # make our gas space or the normal
        guess_space, R = np.linalg.qr(guess_space)
        # the expensive matrix product
        expensive_matrix_product = matrix @ guess_space
        # calculate the rayleigh matrix
        rayleigh_matrix = guess_space.T @ expensive_matrix_product
        # calculate the eigenvalues and eigenvectors of the rayleigh matrix
        eigenvalues, eigenvectors = np.linalg.eig(rayleigh_matrix)
        # sort the eigenvalues and eigenvectors
        sorted_indices = eigenvalues.argsort()
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # calculate the residual vector
        u = guess_space @ eigenvectors[:, i]
        u_a = expensive_matrix_product @ eigenvectors[:, i]
        residual = u - u_a * eigenvalues[i]
        # make a descriptive message and print out the Eigen values
        message = "The length of the eigenvalues is: " + str(len(eigenvalues)) + " and the eigenvalues are: " + str(eigenvalues[:4])
        # print(message)
        # if the norm of the residual is lower than our tolerance come bring the lope as we have found the Eigen pair
        # print out the norm of the residual factor
        print("The norm of the residual is:", np.linalg.norm(residual))
        if np.linalg.norm(residual) < tolerance:
            break
        # if the norm of the residual is not lower than our tolerance, we need to add a new guess vector to our guess space
        else:
            # intact lies our new direction
            new_direction = np.zeros((rows, n_eigval))
            # solved the versed correction equation
            for j in range(rows):
                # if the difference between the rayleigh quotient and the diagonal element is smaller than the tolerance, set the component to zero
                if np.abs(eigenvalues[0] - matrix[j,j]) < tolerance:
                    new_direction[j] = 0
                else:
                    new_direction[j] = residual[j] / (eigenvalues[0] - matrix[j,j])
            # add the new direction to our guess space
            guess_space = np.hstack((guess_space, new_direction))
    return eigenvalues[0]
# find the time that it is taking to generate the hamiltonian
start_generation = time.time()
# generate the hamiltonian
hamiltonian = generation(integrals)
and_generation = time.time()
# fun the time date Davidson da economization takes
start_davidson = time.time()

assert(Davidson(hamiltonian, 2) - -7.8399080148963369 < 1e-10)
# find the time that Davidson digestion takes
end_davidson = time.time()
# find the time that numpy digestion takes
start_numpy = time.time()
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)# thinner
# sort the icon values in assenting order
sorted_indices = eigenvalues.argsort()
eigenvalues = eigenvalues[sorted_indices]
numpy_message = "The length of the eigenvalues is: " + str(len(eigenvalues)) + " and the eigenvalues are: " + str(eigenvalues[:4])
print(numpy_message)
# find the time that numpy digestion takes
end_numpy = time.time()
# print the time differences
print("Generation took:", and_generation - start_generation)
print("Davidson took:", end_davidson - start_davidson)
print("Numpy took:", end_numpy - start_numpy)
assert((end_numpy - start_numpy) > (end_davidson - start_davidson))


