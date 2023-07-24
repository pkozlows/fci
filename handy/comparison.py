import copy
import itertools
import math
import numpy as np

def changing_handy(ci_vector, electrons_in_system, number_of_orbitals, integrals, spin_of_system = 0):
    """takes to integers, representing the number of electrons and orbitals in the system.also takes a tuple with integrals, containing the one electron and two electron integrals for the system.returns the si ci_vector, whatever that means."""
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
    def single_excitation(unoccupied, occupied, n_elec, n_orbs):

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

    # perform 1e excitation to all strings
    def single_replacement(strings, n_elec, n_orbs):
        full_set = set(range(n_orbs))
        return [single_excitation(full_set - set(string), string, n_elec, n_orbs) for string in strings]
    def transform(ci_vector):
        """transforms a configuration interaction ci_vector without using the whole hamiltonian matrices according to the Davidson algorithm. takes the gas ci_vector and returns the transformed ci_vector."""  
        # in nationalize the one particle excitation matrix with np.zeroes
        # for this system come it will have shape (400,6,6), where the first element of the tuple is the dimension of determinant bases and the second and third elements of the tuple are the number of orbitals
        one_particle_matrix = np.zeros((dimension, number_of_orbitals, number_of_orbitals))
        # generate a list of all possible alpha and beta excitations
        alpha_excitations = single_replacement(alpha_strings, number_of_alpha_electrons, number_of_orbitals)
        beta_excitations = single_replacement(beta_strings, number_of_beta_electrons, number_of_orbitals)
        
        # first loop over alpha string replacements
        for alpha_index, replacement_list in enumerate(alpha_excitations):
            for replacement in replacement_list:
                for beta_index in range(len(beta_strings)):
                # first only fill the amendments that will be nonzero with numeral one or numeral negative one
                    i, j = replacement["ij"]
                    ci_vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = replacement["address"] + beta_index * len(beta_strings)
                    one_particle_matrix[one_particle_index, i, j] += replacement["sign"] * ci_vector[ci_vector_index]
        # now loop over debate strings
        for beta_index, replacement_list in enumerate(beta_excitations):
            for replacement in replacement_list:
                for alpha_index in range(len(alpha_strings)):
                    i, j = replacement["ij"]
                    ci_vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = alpha_index + replacement["address"] * len(beta_strings)
                    one_particle_matrix[one_particle_index, i, j] += replacement["sign"] * ci_vector[ci_vector_index]
        # print out the norm of the one particle matrix at this stage with a description
        print("changing_handy", np.linalg.norm(one_particle_matrix))
        # add the original 1e integral and contribution from 1 integral with a delta function \delta_{jk}
        modified_1e_integral = one_electron_integrals - 0.5 * np.einsum("ikkl -> il", two_electron_integrals)
        # now we want to combine the one particle excitation matrix with the relevant part of the two electron integral
        contracted_to_electron = np.einsum('pkl,ijkl->pij', one_particle_matrix, two_electron_integrals)
        # Start from 1e integral transform
        new_ci_vector = np.einsum("pij, ij -> p", one_particle_matrix, modified_1e_integral)
        # now we will be operating on our new ci_vector
        # first lope over the offa replacements
        for alpha_index, replacement_list in enumerate(alpha_excitations):
            for replacement in replacement_list:
                for beta_index in range(len(beta_strings)):
                    i, j = replacement["ij"]
                    ci_vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = replacement["address"] + beta_index * len(beta_strings) 
                    # add the appropriate contribution to our new ci_vector
                    new_ci_vector[ci_vector_index] += 0.5 * replacement["sign"] * contracted_to_electron[one_particle_index, i, j]
        # now loop over beta strings
        for beta_index, replacement_list in enumerate(beta_excitations):
            for replacement in replacement_list:
                for alpha_index in range(len(alpha_strings)):
                    i, j = replacement["ij"]
                    ci_vector_index = alpha_index + beta_index * len(beta_strings)
                    one_particle_index = alpha_index + replacement["address"] * len(beta_strings)
                    # add the appropriate contribution to our new ci_vector
                    new_ci_vector[ci_vector_index] += 0.5 * replacement["sign"] * contracted_to_electron[one_particle_index, i, j]
        return new_ci_vector
    return transform(ci_vector)

