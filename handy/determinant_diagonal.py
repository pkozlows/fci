import itertools

import numpy as np


def diagonal(number_of_orbitals, electrons_in_system, one_electron_integrals, two_electron_integrals, spin_of_system = 0):
    
    """this function calculates the diagonal elements of the Hamiltonian matrix. It is used in the Davidson algorithm to calculate the preconditioner. it takes in the number of orbs and electrons in the system as well as the set of one electron and two electron integrals. It returns the diagonal elements of the Hamiltonian matrix."""
    number_of_alpha_electrons = (electrons_in_system + spin_of_system) // 2
    number_of_beta_electrons = (electrons_in_system - spin_of_system) // 2
    # create all possible alpha strings
    alpha_strings = list(list(string) for string in itertools.combinations(range(number_of_orbitals), number_of_alpha_electrons))
    # create all possible beta strings
    beta_strings = list(list(string) for string in itertools.combinations(range(number_of_orbitals), number_of_beta_electrons))
    dimension = len(alpha_strings) * len(beta_strings)
    # create empty array to store our diagonal elements in
    diagonal_elements = np.zeros(dimension)
    # loop over all our determinants
    for alpha_index in range(len(alpha_strings)):
        for beta_index in range(len(beta_strings)):
            ci_vector_index = alpha_index + beta_index * len(beta_strings)
            one_electron =  

    