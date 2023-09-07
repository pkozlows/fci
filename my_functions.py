import copy
import itertools
import numpy as np
import math

def matrix_davidson(matrix, n_eigval, tolerance = 1e-8):
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

def my_diag(span, electrons, orbs, integrals):
    """takes three integers representing the total spin of the system, the number of elections, and the number of arms.also takes a tuple that has the one electron and two electron integrals for the system.returns an array that represents the takano_set of the full configuration interruption hamiltonian."""
    number_of_alpha_electrons = (electrons + span) // 2
    number_of_beta_electrons = (electrons - span) // 2
    # the one electron integrals
    one_elec_ints = integrals[0]
    # the two electron integrals
    two_elec_ints = integrals[1]
    # generate auto possible orbs
    alpha_orbs = list(list(string) for string in itertools.combinations(range(0, orbs), number_of_alpha_electrons))
    beta_orbs = list(list(string) for string in itertools.combinations(range(0, orbs), number_of_beta_electrons))
    # check that the number of alpha and beta orbs are equal if the spin of the system is zero
    if span == 0:
        assert len(alpha_orbs) == len(beta_orbs)
    diag_list = []
    for alpha in alpha_orbs:
        for beta in beta_orbs:
            # the one electron part
            one_electron = np.einsum('ii->', one_elec_ints[np.ix_(alpha, alpha)]) + np.einsum('ii->', one_elec_ints[np.ix_(beta, beta)])

            # the two electron part
            coulumb = np.einsum('iijj->', two_elec_ints[np.ix_(alpha, alpha, beta, beta)]) + 0.5 * np.einsum('iijj->', two_elec_ints[np.ix_(alpha, alpha, alpha, alpha)]) + 0.5 * np.einsum('iijj->', two_elec_ints[np.ix_(beta, beta, beta, beta)])
            exchange = 0.5 * np.einsum('ijji->', two_elec_ints[np.ix_(alpha, alpha, alpha, alpha)]) + 0.5 * np.einsum('ijji->', two_elec_ints[np.ix_(beta, beta, beta, beta)])
            # add them up
            element = one_electron + coulumb - exchange
            # add the element to the takano_set
            diag_list.append(element)
    return np.array(diag_list)
def handy_davidson(handy_transformer, preconditioner, index, began_search_size, dimension=400, tolerance=1e-4, max_iterations=100):
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
    guess_space = np.eye(dimension, began_search_size) + .000001
    
    for _ in range(max_iterations):
        # print(_)
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
            return eigenvalue
        
        new_directions = np.dot(np.diagflat(1.0 / (eigenvalue - preconditioner)), residue)
        
        # Expand the guess space with new directions while keeping its size limited
        guess_space = np.concatenate((guess_space, np.array([new_directions]).T), axis=1)    
    raise ValueError("Davidson did not converge within the given number of iterations.")
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
    def single_excitation(unoccupied, occupied, n_elec):

        result = []

        for i in occupied:
            result.append({
                "ij": (i, i),
                "sign": 1,
                "address": address_array(occupied)
            })

        for i in list(unoccupied):
            for index, j in enumerate(occupied):
                excited = copy.deepcopy(occupied)
                excited[index] = i
                sign, sorted = face_factor(excited)

                result.append({
                    "ij": (i, j),  # left for excited, right for occupied
                    "sign": sign,
                    "address": address_array(sorted)
                })

        return result
    def single_replacement(strings, n_elec):
        full_set = set(range(number_of_orbitals))
        return [single_excitation(full_set - set(string), string, n_elec) for string in strings]
    def transformer(vector):
        """transforms a configuration interaction vector without using the whole hamiltonian matrices according to the Davidson algorithm. takes the gas vector and returns the transformed vector."""  
        # in nationalize the one particle excitation matrix with np.zeroes
        # for this system come it will have shape (400,6,6), where the first element of the tuple is the dimension of determinant bases and the second and third elements of the tuple are the number of orbitals
        one_particle_matrix = np.zeros((dimension, number_of_orbitals, number_of_orbitals))
        offer_single_replacements = single_replacement(alpha_strings, number_of_alpha_electrons)
        beta_single_replacements = single_replacement(beta_strings, number_of_beta_electrons)    
        # first loop over alpha string replacements
        for alpha_index, alpha_replacement_list in enumerate(offer_single_replacements):
            for replacement in alpha_replacement_list:
                for beta_index in range(len(beta_strings)):
                # first only fill the amendments that will be nonzero with numeral one or numeral negative one
                    vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = replacement["address"] + beta_index * len(beta_strings)
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    one_particle_matrix[one_particle_index, i, j] += replacement["sign"] * vector[vector_index]
        # now loop over debate strings
        for beta_index, beta_replacement_list in enumerate(beta_single_replacements):
            for replacement in beta_replacement_list:
                for alpha_index in range(len(alpha_strings)):
                    vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = alpha_index + replacement["address"] * len(beta_strings)
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    one_particle_matrix[one_particle_index, i, j] += replacement["sign"] * vector[vector_index]
        # add the original 1e integral and contribution from 1 integral with a delta function \delta_{jk}
        modified_1e_integral = one_electron_integrals - 0.5 * np.einsum("ikkl -> il", two_electron_integrals)
        # now we want to combine the one particle excitation matrix with the relevant part of the two electron integral
        contracted_to_electron = np.einsum('pkl,ijkl->pij', one_particle_matrix, two_electron_integrals)
        # Start from 1e integral transform
        new_ci_vector = np.einsum("pij, ij -> p", one_particle_matrix, modified_1e_integral)
        # now we will be operating on our new vector
        # first lope over the offa replacements
        for alpha_index, alpha_replacement_list in enumerate(offer_single_replacements):
            for replacement in alpha_replacement_list:
                for beta_index in range(len(beta_strings)):
                    vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = replacement["address"] + beta_index * len(beta_strings) 
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    # add the appropriate contribution to our new vector
                    new_ci_vector[vector_index] += 0.5 * replacement["sign"] * contracted_to_electron[one_particle_index, i, j]
        # now loop over beta strings
        for beta_index, beta_replacement_list in enumerate(beta_single_replacements):
            for replacement in beta_replacement_list:
                for alpha_index in range(len(alpha_strings)):
                    vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = alpha_index + replacement["address"] * len(beta_strings)
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    # add the appropriate contribution to our new vector
                    new_ci_vector[vector_index] += 0.5 * replacement["sign"] * contracted_to_electron[one_particle_index, i, j]
        return new_ci_vector
    return transformer

