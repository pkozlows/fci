import copy
import itertools
import math
import numpy as np
import signal
import time
import cProfile
from handy.mentor_handy import knowles_handy_full_ci_transformer

from diag import diagonal

integrals = (np.load("h1e.npy"), np.load("h2e.npy"))

def handy_transformer(electrons_in_system, number_of_orbitals, integrals, spin_of_system = 0):
    """takes to integers, representing the number of electrons and orbitals in the system.also takes a tuple with integrals, containing the one electron and two electron integrals for the system.returns the si vector, whatever that means."""
    # find the number of alpha and better electrons in the system
    one_electron_integrals = integrals[0]
    two_electron_integrals = integrals[1]
    number_of_alpha_electrons = (electrons_in_system + spin_of_system) // 2
    number_of_beta_electrons = (electrons_in_system - spin_of_system) // 2
    # create all possible alpha strings
    alpha_strings = list(list(string) for string in itertools.combinations(range(number_of_orbitals), number_of_alpha_electrons))
    # create all possible beta strings
    beta_strings = list(list(string) for string in itertools.combinations(range(number_of_orbitals), number_of_beta_electrons))
    dimension = len(alpha_strings) * len(beta_strings)
    def face_factor(excited_string):
        """takes one unsorted excited string. Returns a integer that represents the face_factor for canting the strings into canonical order."""
        # make a deep copy of the excited list to be later sorted
        sorted = copy.deepcopy(excited_string)
        # initialize the number of swaps needed to zero
        swaps = 0
        i = 0
        while i < len(excited_string) - 1:
            if sorted[i] > sorted[i+1]:
                temp = sorted[i+1]
                sorted[i+1] = sorted[i]
                sorted[i] = temp
                swaps += 1
                i = 0
            else:
                i += 1
        # return the face_factor and the sorted list
        return (-1)**swaps, sorted
    def Z(electron_index, orbital_index, number_of_electrons):
        """takes 2 integers, representing the electron_index and orbital_index numbers. this a function that was introduced in the candy paper from 1984.returns the Z factor for the given numbers."""
        if electron_index == number_of_electrons:
            return orbital_index - number_of_electrons
        else:
            return sum([math.comb(m, number_of_electrons - electron_index) - math.comb(m - 1, number_of_electrons - electron_index - 1) for m in range(number_of_orbitals - orbital_index + 1, number_of_orbitals - electron_index + 1)])
    def address_array(string):
        assert(len(string)) == 3
        return sum([Z(electron_index + 1, orbital + 1, len(string)) for electron_index, orbital in enumerate(string)])
    def single_replacement(unexcited_string):
        """takes an unexcited string and returns a list of strings with one excitation"""
        # look at a very orbital in the string end enervate all excitations from it
        # in naturalize a big list for ate excitations
        big_excitations = []
        for i, orbital in enumerate(unexcited_string):
            # look at all orbitals different than the current one that are not already in the list
            # in naturalize a list for all these little excitations
            little_excitations = []
            for j in range(number_of_orbitals):
                excited_string = copy.deepcopy(unexcited_string)
                if j not in unexcited_string:
                    # create a new string with the excitation
                    excited_string[i] = j
                    # attend the possible excited strings to a list
                    little_excitations.append(excited_string)
            # now append ofthe little excitations to a big list
            big_excitations.append(little_excitations)
        return big_excitations
    def replacement_list(original_string):
        """takes one original string and returns a list of tuples. The tuples contain the address of the replaced_string, the face_factor, and the value of the matrix element."""
        # in nationalized global alpha and beta lists
        replacements = []
        # first append all of the diagonal elements
        for orbital in original_string:
            replacements.append({"address": address_array(original_string), "sign": 1, "ij": (orbital, orbital)})      
        # We may generate all the matrix elements by constructing two lists (one for Q, OIIC for /3) for each K (i.e. +O+p) of ail single rtpfacements. Ed& entry in the fist contains three integers- the lexical address of +I,‘, the numerical value of the matrix element which is +I or --I, and ii_. this will be contained in a tuple, like (address, sign, excitation (ij for i->j)))
        all_orbitals = single_replacement(original_string)
        for replaced_orbital_list in all_orbitals:
            for replaced_string in replaced_orbital_list:
                # I want to find the unexcited orbital that is in the after replaced_string comma but not in this new replaced_string
                ground = [orbital for orbital in original_string if orbital not in replaced_string]
                excited = [orbital for orbital in replaced_string if orbital not in original_string]
                sign, sorted = face_factor(replaced_string)
                replacements.append( {"address": address_array(sorted), "sign": sign, "ij": (ground[0], excited[0])} )
        return replacements
    def transformer(vector):
        """transforms a configuration interaction vector without using the whole hamiltonian matrices according to the Davidson algorithm. takes the gas vector and returns the transformed vector."""  
        # in nationalize the one particle excitation matrix with np.zeroes
        # for this system come it will have shape (400,6,6), where the first element of the tuple is the dimension of determinant bases and the second and third elements of the tuple are the number of orbitals
        one_particle_matrix = np.zeros((dimension, number_of_orbitals, number_of_orbitals))
        # first loop over alpha string replacements
        for alpha_index, alpha_string in enumerate(alpha_strings):
            for replacement in replacement_list(alpha_string):
                for beta_index in range(len(beta_strings)):
                # first only fill the amendments that will be nonzero with numeral one or numeral negative one
                    vector_index = replacement["address"] + beta_index * len(beta_strings)
                    one_particle_index = alpha_index + beta_index * len(beta_strings)
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    one_particle_matrix[one_particle_index, i, j] += np.real(replacement["sign"] * vector[vector_index])
        # now loop over debate strings
        for beta_index, beta_string in enumerate(beta_strings):
            for replacement in replacement_list(beta_string):
                for alpha_index in range(len(alpha_strings)):
                    vector_index = replacement["address"] + alpha_index * len(beta_strings)
                    one_particle_index = beta_index + alpha_index * len(beta_strings)
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    one_particle_matrix[one_particle_index, i, j] += np.real(replacement["sign"] * vector[vector_index])
        # add the original 1e integral and contribution from 1 integral with a delta function \delta_{jk}
        modified_1e_integral = one_electron_integrals - 0.5 * np.einsum("ikkl -> il", two_electron_integrals)
        # now we want to combine the one particle excitation matrix with the relevant part of the two electron integral
        contracted_to_electron = np.einsum('pkl,ijkl->pij', one_particle_matrix, two_electron_integrals)
        # Start from 1e integral transform
        new_ci_vector = np.einsum("pij, ij -> p", one_particle_matrix, modified_1e_integral)
        # now we will be operating on our new vector
        # first lope over the offa replacements
        for alpha_index, alpha_string in enumerate(alpha_strings):
            for replacement in replacement_list(alpha_string):
                for beta_index in range(len(beta_strings)):
                    one_particle_index = replacement["address"] + beta_index * len(beta_strings)
                    vector_index = alpha_index + beta_index * len(beta_strings) 
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    # add the appropriate contribution to our new vector
                    new_ci_vector[vector_index] += 0.5 * contracted_to_electron[one_particle_index, i, j]
        # now loop over beta strings
        for beta_index, beta_string in enumerate(beta_strings):
            for replacement in replacement_list(beta_string):
                for alpha_index in range(len(alpha_strings)):
                    one_particle_index = replacement["address"] + alpha_index * len(beta_strings)
                    vector_index = beta_index + alpha_index * len(beta_strings)
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    # add the appropriate contribution to our new vector
                    new_ci_vector[vector_index] += 0.5 * contracted_to_electron[one_particle_index, i, j]
        return new_ci_vector
    return transformer
# check if the norm of my handy transformer operating on a configuration interaction vector is the same as the norm from another transformer function
def check_transformer(transformer, other_transformer, dimension = 400):
    """checks if the norm of my handy transformer operating on a configuration interaction vector is the same as the norm from another transformer function"""
    # generate a random vector we has a length of 1 and the dimension on the other axis
    vector = np.random.rand(dimension)
    # check if the norm of my handy transformer operating on a configuration interaction vector is the same as the norm from another transformer function
    assert np.isclose(np.linalg.norm(transformer(vector)), np.linalg.norm(other_transformer(vector)))
    print("The norm of the vector is the same for both transformers")
    return None
# check_transformer(transformer, other_transformer)
def davidson_diagonalization(transformer,
                             diagonal,
                             eigenvalue_index,
                             start_search_dim,
                             n_dim,
                             residue_tol=1e-5,
                             max_iter=1000):

    search_space = np.eye(n_dim, start_search_dim) + 0.01

    for iter in range(max_iter):
        print(iter)
        # perform QR decomposition to make sure the column vectors are orthonormal
        orthonormal_subspace, upper_triangular = np.linalg.qr(search_space)

        M = orthonormal_subspace.shape[1]

        Ab_i = np.zeros((n_dim, M))

        for i in range(M):

            Ab_i[:, i] = transformer(orthonormal_subspace[:, i])

        interaction_matrix = np.dot(orthonormal_subspace.T, Ab_i)
        eigs, eigvecs = np.linalg.eig(interaction_matrix)

        sorted_indices = eigs.argsort()
        eig = eigs[sorted_indices[eigenvalue_index]]
        eigvec = eigvecs[:, sorted_indices[eigenvalue_index]]

        residue = np.dot(Ab_i, eigvec) - eig * np.dot(orthonormal_subspace, eigvec)
        print(np.linalg.norm(residue))
        if np.linalg.norm(residue) < residue_tol:
            return eig, eigvec

        xi = np.dot(np.diagflat(1.0 / (eig - diagonal)), residue)

        np.eye(n_dim) - np.einsum('ij, kj -> jik', orthonormal_subspace, orthonormal_subspace)

        search_space = np.concatenate((orthonormal_subspace, np.array([xi]).T), axis=1)

    raise Exception("Davidson diagonaliztion failed")
def Davidson(handy_transformer, preconditioner, index, began_search_size, dimension=400, tolerance=1e-5, max_iterations=100):
    """
    Implements the Davidson algorithm to approximate the lowest eigenvalues and eigenvectors of a matrix.
    
    Parameters:
    - handy_transformer: A function that multiplies the matrix by a vector.
    - began_search_size: Number of desired eigenvalues.
    - preconditioner: Diagonal preconditioner for Davidson corrections.
    - dimension: Dimension of the matrix.
    - tolerance: Convergence criteria.
    - max_iterations: Maximum number of Davidson iterations.

    Returns:
    - Approximate eigenvalues of the matrix.
    """
    
    # Initial guess space
    guess_space = np.eye(dimension, began_search_size) + .1
    
    for _ in range(max_iterations):
        print(_)
        # Orthogonalize guess space
        guess_space, _ = np.linalg.qr(guess_space)
        # make a transform space that is the same dimension as our guess space
        transformed_space = np.zeros((dimension, guess_space.shape[1]))
        # Apply the matrix-vector product using handy_transformer for all vectors in guess_space
        for i in range(guess_space.shape[1]):
            transformed_space[:, i] = handy_transformer(guess_space[:, i])
            
        
        # Compute the Rayleigh quotient
        rayleigh_matrix = guess_space.T @ transformed_space
        
        # Get eigenvalues and eigenvectors of Rayleigh quotient
        eigenvalues, eigenvectors = np.linalg.eig(rayleigh_matrix)
        
        # Sort the eigenvalues
        sorted_indices = eigenvalues.argsort()
        # get the again system that I want
        eigenvalue = eigenvalues[sorted_indices[index]]
        eigenvector = eigenvectors[:, sorted_indices[index]]
        
        
        # Calculate residue
        residue = np.dot(transformed_space, eigenvector) - eigenvalue * np.dot(guess_space, eigenvector)

        # Convergence check
        if np.linalg.norm(residue) < tolerance:
            return eigenvalue, eigenvector
        
        new_directions = np.dot(np.diagflat(1.0 / (eigenvalue - preconditioner)), residue)
        
        # Expand the guess space with new directions while keeping its size limited
        guess_space = np.concatenate((guess_space, np.array([new_directions]).T), axis=1)    
    raise ValueError("Davidson did not converge within the given number of iterations.")


# # fun the time date Davidson da economization takes
# start_davidson = time.time()
# print(davidson_diagonalization(handy_transformer(6, 6, integrals), diagonal(0, 6, 6, integrals), 0, 2, 400))
def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")

def profile_with_timeout(func, timeout):
    """Profile a function with a timeout."""
    # Register the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Set the alarm
    signal.alarm(timeout)
    try:
        cProfile.run(func, sort="cumtime")
    except TimeoutError:
        print(f"Function was profiled for {timeout} seconds before timing out.")
# create the transformers
mentor_transformer = knowles_handy_full_ci_transformer(integrals[0], integrals[1], 6)
my_transformer = handy_transformer(6, 6, integrals)
# create the diagonal

mentor_davison = davidson_diagonalization(transformer, diagonal, 0, 2, 400)
my_davidson = Davidson(handy_transformer(6, 6, integrals), diagonal(0, 6, 6, integrals), 0, 2)
print(davidson_diagonalization(handy_transformer(6, 6, integrals), diagonal(0, 6, 6, integrals), 0, 2, 400))   
# Use it like this:
# profile_with_timeout("davidson_diagonalization(handy_transformer(6, 6, integrals), diagonal(0, 6, 6, integrals), 0, 2, 400)", 10)  # Runs the profiler on your_function() for 10 seconds
# profile_with_timeout("Davidson(handy_transformer(6, 6, integrals), 2, diagonal(0, 6, 6, integrals))", 10)  # Runs the profiler on your_function() for 10 seconds

# cProfile.run('Davidson(handy_transformer(6, 6, integrals), diagonal(0, 6, 6, integrals), 0, 2)')
# assert(Davidson(knowles_handy_full_ci_transformer(one_electron_integrals=), 1) - -7.8399080148963369 < 1e-10)
# # find the time that Davidson digestion takes
# end_davidson = time.time()




