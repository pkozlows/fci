import copy
import itertools
import math
import numpy as np

def handy_transformer(vector, electrons_in_system, number_of_orbitals, integrals, spin_of_system = 0):
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
    def transform(vector):
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
                    one_particle_matrix[one_particle_index, i, j] += replacement["sign"] * vector[vector_index]
        # now loop over debate strings
        for beta_index, beta_string in enumerate(beta_strings):
            for replacement in replacement_list(beta_string):
                for alpha_index in range(len(alpha_strings)):
                    vector_index = replacement["address"] + alpha_index * len(beta_strings)
                    one_particle_index = beta_index + alpha_index * len(beta_strings)
                    i, j = replacement["ij"][0], replacement["ij"][1]
                    one_particle_matrix[one_particle_index, i, j] += replacement["sign"] * vector[vector_index]
        # print out the norm of the one particle matrix at this stage with a description
        print("handy_initial", np.linalg.norm(one_particle_matrix))
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
    return transform(vector)

