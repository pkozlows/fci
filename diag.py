import itertools

import numpy as np


def mentor_diag(span, electrons, integrals):
    """Compute the diagonal of the full configuration interaction Hamiltonian."""
    n_rows, n_cols = integrals[0].shape

    n_orbs = n_rows

    assert(n_rows == n_cols)
    assert(np.all(np.array(integrals[1].shape) == n_orbs))

    n_alpha = (electrons + span) // 2
    n_beta = (electrons - span) // 2

    alpha_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    n_dim = len(alpha_combinations) * len(beta_combinations)

    diagonal = []

    for i in range(n_dim):
        i_alpha_combination = alpha_combinations[i % len(beta_combinations)]
        i_beta_combination = beta_combinations[i // len(beta_combinations)]

        one_electron_part = (
            np.einsum("ii->", integrals[0][np.ix_(i_alpha_combination, i_alpha_combination)]) +
            np.einsum("ii->", integrals[0][np.ix_(i_beta_combination, i_beta_combination)])
        )
        coulomb_part = (
            np.einsum("iijj->", integrals[1][np.ix_(i_alpha_combination, i_alpha_combination, i_beta_combination, i_beta_combination)]) +
            0.5 * np.einsum("iijj->", integrals[1][np.ix_(i_alpha_combination, i_alpha_combination, i_alpha_combination, i_alpha_combination)]) +
            0.5 * np.einsum("iijj->", integrals[1][np.ix_(i_beta_combination, i_beta_combination, i_beta_combination, i_beta_combination)])
        )
        exchange_part = (
            0.5 * np.einsum("ijji->", integrals[1][np.ix_(i_alpha_combination, i_alpha_combination, i_alpha_combination, i_alpha_combination)]) +
            0.5 * np.einsum("ijji->", integrals[1][np.ix_(i_beta_combination, i_beta_combination, i_beta_combination, i_beta_combination)])
        )
        element = one_electron_part + coulomb_part - exchange_part
        diagonal.append(element)

    return diagonal

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

    