import itertools
import math
import numpy as np

def handy(elec_in_system, orbs, intog, spin_of_system = 0):
    """takes to integers, representing the number of electrons and orbitals in the system.also takes a tuple with integrals, containing the one electron and two electron integrals for the system.returns the si vector, whatever that means."""
    # find the number of alpha and better electrons in the system
    number_of_alpha_electrons = (elec_in_system + spin_of_system) // 2
    number_of_beta_electrons = (elec_in_system - spin_of_system) // 2
    # create all possible alpha strings
    alpha_strings = list(list(string) for string in itertools.combinations(range(0, orbs), number_of_alpha_electrons))
    # create all possible beta strings
    beta_strings = list(list(string) for string in itertools.combinations(range(0, orbs), number_of_beta_electrons))
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
            for j in range(orbs):
                excited_string = unexcited_string.copy()
                if j != orbital and j not in unexcited_string:
                    # create a new string with the excitation
                    excited_string[i] = j
                    # attend the possible excited strings to a list
                    little_excitations.append(excited_string)
            # now append ofthe little excitations to a big list
            big_excitations.append(little_excitations)
        return big_excitations
    def Z(electron, orbital, number_of_electrons, number_of_orbitals):
        """takes 4 integers, representing the electron and orbital numbers, and the number of electrons and orbitals.this a function that was introduced in the candy paper from 1984.returns the Z factor for the given numbers."""
        if electron == number_of_electrons:
            return orbital - number_of_electrons
        else:
            return sum([math.comb(m, number_of_electrons - electron) - math.comb(m - 1, number_of_electrons - electron - 1)
                    for m in range(number_of_orbitals - orbital + 1, number_of_orbitals - electron + 1)])
    # apply address to a configuration
    def address_array(string, number_of_electrons, number_of_orbitals):
    # +1 is the conversion between python indexing (start with 0) and normal indexing (start with 1)
    # Haiya Starting from 0 makes life easier, e.g. the indexing of tensor product
        return sum([Z(electron + 1, orbital + 1, number_of_electrons, number_of_orbitals) for electron, orbital in enumerate(string)])
    def adress(first_string, second_string, spin):
        """takes two strings and and either and integer of 1 and the citing that the two strings are of spin alpha or a integer of 0, and the citing that they are two string of spin peter. returns their address as a tuple."""
        if spin == 1:
            return (alpha_strings.index(first_string), alpha_strings.index(second_string))
        elif spin == 0:
            return (beta_strings.index(first_string), beta_strings.index(second_string))
    def face_factor(unexcited, excited):
        """takes two list. The first one is an unexcited string, the second one is a list of all possible excitations from that string. it does not come as sorted. Returns a integer that represents the face_factor for canting the strings into maximum coincidence."""
        # check that the first string is already sorted
        assert(unexcited sorted(unexcited))
        # find the difference between the excited string and the unexcited string
        difference = [orbital for orbital in excited if orbital not in unexcited]
        # initialize a sorted list of the excited string
        sorted_excited = sorted(excited)
        # initialize the number of swaps needed to zero
        swaps = 0
        swaps += sorted_excited.index(difference[0])
        # return the face_factor
        return (-1)**swaps
    def replacement_list(alpha_strings, beta_strings):
        """takes two lists of strings and returns two lists of tuples. The first list of tuples contains all possible replacements for the alpha strings. The second list of tuples contains all possible replacements for the beta strings. The tuples contain the address of the string, the face_factor, and the value of the matrix element."""
        # in nationalized global alpha and beta lists
        global_alfa = []
        global_beater = []
        # iterate over all unexcited Pearse
        for  alpha_string in alpha_strings:
            for beta_string in beta_strings:
                # We may generate all the matrix elements by constructing two lists (one for Q, OIIC for /3) for each K (i.e. +O+p) of ail single rtpfacements. Ed& entry in the fist contains three integers- the lexical address of +I,‘, the numerical value of the matrix element which is +I or --I, and ii_. this will be contained in a tuple, like (address, face_a_factor, excitation (ij for i->j)))
                # first make a list for the alpha replacements
                alpha_replacements = single_replacement(alpha_string)
                alpha_list = []
                for replaced in alpha_replacements:
                    for string in replaced:
                        # I want to find the unexcited orbital that is in the after string comma but not in this new string
                        ground = [orbital for orbital in alpha_string if orbital not in string]
                        assert(len(ground) == 1)
                        excited = [orbital for orbital in string if orbital not in alpha_string]
                        assert(len(excited) == 1)
                        alpha_list.append((adress(alpha_string, sorted(string), 0), face_factor(alpha_string, string), (ground[0], excited[0])))
                global_alfa.append(alpha_list)
                # then make a list for the beta replacements
                beta_replacements = single_replacement(beta_string)
                beater_list = []
                for replaced in beta_replacements:
                    for string in replaced:
                        # I want to find the unexcited orbital that is in the after string comma but not in this new string
                        ground = [orbital for orbital in beta_string if orbital not in string]
                        assert(len(ground) == 1)
                        excited = [orbital for orbital in string if orbital not in beta_string]
                        assert(len(excited) == 1) 
                        beater_list.append((adress(beta_string, sorted(string), 0), face_factor(beta_string, string), (ground[0], excited[0])))
                global_beater.append(beater_list)
        return (global_alfa, global_beater)
    return replacement_list(alpha_strings, beta_strings)

print(handy(6, 6, (np.load("h1e.npy"), np.load("h2e.npy"))))

