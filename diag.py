import itertools

import numpy as np


def diagonal(span, electrons, orbs, integrals):
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
print(diagonal(0, 6, 6, (np.load("h1e.npy"), np.load("h2e.npy"))))

    