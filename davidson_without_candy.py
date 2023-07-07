import time
import numpy as np
from main import generation, integrals


def Davidson(taster, transformer, preconditioner, n_eig, tolerance = 1e-8, matrix = None):
    """takes the transformer function of handy, the preconditioner function of handy, the number of eigenvalues to be found, the tolerance for the algorithm to converge, and the matrix to be diagonalized.returns the size of the agen system to be found."""
    # starred by assuming that we will be using the handy implementation
    handy = True
    # checked if we are preceding with the handy intimation or not
    if matrix == None:
        handy = False
        # check that the matrix is symmetric
        assert(np.allclose(matrix, matrix.T))
        rows, cols = matrix.shape
    # make our initial guess the hf reference eigenvector
    guess_space = np.eye(rows, n_eigval)
    for i in range(rows // 2):
        # make our gas space or the normal
        guess_space, R = np.linalg.qr(guess_space)
        # calculate the rayleigh matrix
        rayleigh_matrix = guess_space.T @ matrix @ guess_space
        # calculate the eigenvalues and eigenvectors of the rayleigh matrix
        eigenvalues, eigenvectors = np.linalg.eig(rayleigh_matrix)
        # sort the eigenvalues and eigenvectors
        sorted_indices = eigenvalues.argsort()
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # calculate the residual vector
        residual = matrix @ guess_space @ eigenvectors[:, 0] - eigenvalues[0] * guess_space @ eigenvectors[:, 0]
        # if the norm of the residual is lower than our tolerance come bring the lope as we have found the Eigen pair
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

matrix = generation(integrals)
# fun the time date Davidson da economization takes
start_davidson_without_handy = time.time()
assert(Davidson(matrix, 1) - -7.8399080148963369 < 1e-10)
# find the time that Davidson digestion takes
end_davidson_without_handy = time.time()
# find the time that numpy digestion takes
start_numpy = time.time()
eigenvalues, eigenvectors = np.linalg.eig(matrix)
# find the time that numpy digestion takes
end_numpy = time.time()
assert((end_numpy - start_numpy) > (end_davidson_without_handy - start_davidson_without_handy))



    


                       
#     # initialize a matrix of zeros to place our initial guess in
#     guess_space = np.zeros((rows, cols))
#     for m in range(rows // 2):
#         # check if this is our first iteration
#         if m == 0:
#             # if it is, normalize the guess
#             guess_space[:, 0] = guess[:, 0] / np.linalg.norm(guess[:, 0])
#             # calculate the expensive matrix vector product
#             matrix_vector_product = np.einsum('ij,j->i', matrix, guess_space[:, 0])
#             # calculate the rayleigh quotient
#             rayleigh_quotient = guess_space[:, 0].T @ matrix_vector_product
#             # calculate the residual
#             residual = matrix_vector_product - rayleigh_quotient * guess_space[:, 0]
#             # Approximately solve the residue correction equation. By using a diagonal approximation of the matrix, get a clear expression for a new guess
#             for i in range(rows):
#                 # if rayleigh_quotient - matrix[i, i] is smaller than it resold value, said the component to zero
#                 if np.abs(rayleigh_quotient - matrix[i, i]) < 1e-16:
#                     guess_space[i, 1] = 0
#                 else:
#                     guess_space[i, 1] = residual[i] / (rayleigh_quotient - matrix[i, i])
#             print(guess_space[:, :2])
#             #  calculate the subspace matrix for the iteration
#             subspace_make_checks = guess_space[:, :m+1].T @ matrix @ guess_space[:, :m+1]
#             # find the eigenvalues and eigenvectors of the subspace hamiltonian
#             THETA, U = np.linalg.eigh(subspace_make_checks)
#             # calculate the new residual vector in pieces
#             print(U)
#         else:
#             # Compute the orthogonal projection
#             I = np.eye(guess_space.shape[0])  # Identity matrix of appropriate size
#             tensor_product = I - np.einsum('ij,jk->ik', guess_space[:, :m], guess_space[:, :m].T)
#             print(tensor_product)
#             print(tensor_product.shape)
#             print(guess_space[:, m-1])
#             orthogonal_complement = np.einsum('ij,j->i', tensor_product, guess_space[:, m-1])
#             # Normalize the new guess
#             guess_space[:, m] = orthogonal_complement / np.linalg.norm(orthogonal_complement)
#             # Calculate the matrix vector product and added to the transformed subspace
#             if m == 1:
#                 matrix_vector_product = np.einsum('ij,j->i', matrix, guess_space[:, m])
#             else:
#                 matrix_vector_product = np.einsum('ij,jk->ik', matrix, guess_space[:, m])
                
#             # expand the transformed subspace to include our new guest
#             if m == 1:
#                 # initialize the transform space
#                 transformed_space = np.zeros((rows, cols))    
#             transformed_space[:, m-1] = matrix_vector_product
#             # calculate our new subspace matrix
#             subspace_matrix = np.einsum('ij,jk,kl->il', guess_space[:, :m+1].T, matrix, guess_space[:, :m+1])
#             # solve the subspace eigenvalue problem
#             THETA, U = np.linalg.eigh(subspace_matrix)
#             # start the process of calculating the new residual vector
#             residual = np.einsum('ij,j->i', guess_space[:, :m+1], U[:, 0])
#             transformed_residual = np.einsum('ij,j->i', transformed_space[:, :m+1], U[:, 0])
#             # calculate the actual residual vector
#             actual_residual = transformed_residual - THETA[0] * residual
#             # if the numb of the residual vector is lower than that resold the algorithm has converged and returned the the west I can par
#             if np.linalg.norm(actual_residual) < tolerance:
#                 return THETA, U
#             # solve the residue correction equation to generate a new guess
#             for i in range(rows):
#                 if np.abs(THETA[0] - matrix[i, i]) < 1e-16:
#                     guess_space[i, m+1] = 0
#                 else:
#                     guess_space[i, m+1] = actual_residual[i] / (THETA[0] - matrix[i, i])
           

#         #     # for j in range(0, n_eigval):
#         #     #     guess_space[:, j] = guess[:, j] / np.linalg.norm(guess[:, j])
#         #     #     theta_old = 0
#         # else:
#         #     # Project our new guess onto the orthogonal complement of the old guess subspace
#         #     for i in range(rows):
#         #         new_guess[i, :] = guess_space[i, m-1] - np.einsum('->', ) 
#         # guess_space[:, :m], R = np.linalg.qr(guess_space[:, :m])
#         # # create the subspace hamiltonian
#         # H = guess_space[:, :m].T @ matrix @ guess_space[:, :m]
#         # # find the eigenvalues and eigenvectors of the subspace hamiltonian
#         # THETA, U = np.linalg.eigh(H)
#         # sorted_i_can_values = np.argsort(THETA)
#         # theta = THETA[sorted_i_can_values]
#         # u = U[:, sorted_i_can_values]
#         # for j in range(0, n_eigval):
            
            

                
            
            

        
