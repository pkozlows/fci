import copy
import itertools
import math
import numpy as np

def handy(electrons_in_system, number_of_orbitals, intog, spin_of_system = 0):
    """takes to integers, representing the number of electrons and orbitals in the system.also takes a tuple with integrals, containing the one electron and two electron integrals for the system.returns the si vector, whatever that means."""
    # find the number of alpha and better electrons in the system
    number_of_alpha_electrons = (electrons_in_system + spin_of_system) // 2
    number_of_beta_electrons = (electrons_in_system - spin_of_system) // 2
    # create all possible alpha strings
    alpha_strings = list(list(string) for string in itertools.combinations(range(0, number_of_orbitals), number_of_alpha_electrons))
    # create all possible beta strings
    beta_strings = list(list(string) for string in itertools.combinations(range(0, number_of_orbitals), number_of_beta_electrons))
    dimension = len(alpha_strings) * len(beta_strings)
    def single_replacement(unexcited_string):
        """takes a string and returns a list of strings with one excitation"""
        # look at a very orbital in the string end enervate all excitations from it
        # in naturalize a big list for ate excitations
        big_excitations = []
        for i, orbital in enumerate(unexcited_string):
            # look at all orbitals different than the current one that are not already in the list
            # in naturalize a list for all these little excitations
            little_excitations = []
            for j in range(number_of_orbitals):
                excited_string = unexcited_string.copy()
                if j != orbital and j not in unexcited_string:
                    # create a new string with the excitation
                    excited_string[i] = j
                    # attend the possible excited strings to a list
                    little_excitations.append(excited_string)
            # now append ofthe little excitations to a big list
            big_excitations.append(little_excitations)
        return big_excitations
    def Z(electron_index, orbital_number, number_of_electrons):
        """takes 4 integers, representing the electron_index and orbital_number numbers, and the number of electrons and.this a function that was introduced in the candy paper from 1984.returns the Z factor for the given numbers."""
        if electron_index == number_of_electrons:
            return orbital_number - number_of_electrons
        else:
            return sum([math.comb(m, number_of_electrons - electron_index) - math.comb(m - 1, number_of_electrons - electron_index - 1) for m in range(number_of_orbitals - orbital_number + 1, number_of_orbitals - electron_index + 1)])
    # apply address to a configuration
    def address_array(string):
        return sum([Z(electron_index + 1, orbital_number + 1, len(string)) for electron_index, orbital_number in enumerate(string)])
    def face_factor(excited):
        """takes two list. The first one is an unexcited string, the second one is a list of all possible excitations from that string. it does not come as sorted. Returns a integer that represents the face_factor for canting the strings into maximum coincidence."""
        # make a deep copy of the excited list to be later sorted
        sorted = copy.deepcopy(excited)
        # initialize the number of swaps needed to zero
        swaps = 0
        i = 0
        while i < len(excited) - 1:
            if sorted[i] > sorted[i+1]:
                temp = sorted[i+1]
                sorted[i+1] = sorted[i]
                sorted[i] = temp
                swaps += 1
                i = 0
            else:
                i += 1
        # return the face_factor
        return (-1)**swaps
    def replacement_list(original_string):
        """takes two lists of strings and returns two lists of tuples. The first list of tuples contains all possible replacements for the alpha strings. The second list of tuples contains all possible replacements for the beta strings. The tuples contain the address of the replaced_string, the face_factor, and the value of the matrix element."""
        # in nationalized global alpha and beta lists
        replacements = []
        # first append all of the diagonal elements
        for orbital in original_string:
            replacements.append((address_array(original_string), 1, (orbital, orbital)))          
        # We may generate all the matrix elements by constructing two lists (one for Q, OIIC for /3) for each K (i.e. +O+p) of ail single rtpfacements. Ed& entry in the fist contains three integers- the lexical address of +I,‘, the numerical value of the matrix element which is +I or --I, and ii_. this will be contained in a tuple, like (address, face_a_factor, excitation (ij for i->j)))
        # first make a list for the alpha replacements
        all_orbitals = single_replacement(original_string)
        for replaced_orbital_list in all_orbitals:
            for replaced_string in replaced_orbital_list:
                # I want to find the unexcited orbital that is in the after replaced_string comma but not in this new replaced_string
                ground = [orbital for orbital in original_string if orbital not in replaced_string]
                excited = [orbital for orbital in replaced_string if orbital not in original_string]
                replacements.append((address_array(sorted(replaced_string)), face_factor(replaced_string), (ground[0], excited[0])))
        return replacements
    # make the replacement list for the alpha and peter orbs
    # initialize the two replacement list
    alter_replacement_list = []
    beta_replacement_list = []
    for string in alpha_strings:
        alter_replacement_list.append(replacement_list(string))
    for string in beta_strings:
        beta_replacement_list.append(replacement_list(string))        
    return alter_replacement_list, beta_replacement_list

print(handy(6, 6, (np.load("h1e.npy"), np.load("h2e.npy"))))

