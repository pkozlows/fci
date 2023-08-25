import itertools
import copy
import numpy as np
import math

# Simple bubble sort to get


def sort_and_sign(listable):
    sorted = copy.deepcopy(listable)
    sign = 0

    i = 0
    while i < len(listable) - 1:
        if sorted[i] > sorted[i+1]:
            temp = sorted[i+1]
            sorted[i+1] = sorted[i]
            sorted[i] = temp
            sign += 1
            i = 0
        else:
            i += 1

    return math.pow(-1, sign % 2), sorted

# A function defined in KH paper
def Z(k, l, n_elec, n_orbs):
    assert(n_orbs == 6)
    assert(n_elec == 3)
    if k == n_elec:
        return l - n_elec
    else:
        return sum([math.comb(m, n_elec - k) - math.comb(m - 1, n_elec - k - 1)
                    for m in range(n_orbs - l + 1, n_orbs - k + 1)])

# apply address to a configuration
def address_array(orbital_list, n_elec, n_orbs):
    assert(n_elec == len(orbital_list))
    assert(len(orbital_list) == 3)
    assert(n_orbs == 6)
    # +1 is the conversion between python indexing (start with 0) and normal indexing (start with 1)
    # Haiya Starting from 0 makes life easier, e.g. the indexing of tensor product
    return sum([Z(elec_index + 1, orbital + 1, n_elec, n_orbs) for elec_index, orbital in enumerate(orbital_list)])

# generate list of 1e excitation from original string (occupied)
def single_excitation(unoccupied, occupied, n_elec, n_orbs):

    result = []

    for i in occupied:
        result.append({
            "ij": (i, i),
            "sign": 1,
            "det_index": address_array(occupied, n_elec, n_orbs)
        })

    for i in list(unoccupied):
        for index, j in enumerate(occupied):
            excited = copy.deepcopy(occupied)
            excited[index] = i
            sign, sorted = sort_and_sign(excited)

            result.append({
                "ij": (i, j),  # left for excited, right for occupied
                "sign": sign,
                "det_index": address_array(sorted, n_elec, n_orbs)
            })

    return result

# perform 1e excitation to all strings
def single_replacement(strings, n_elec, n_orbs):
    full_set = set(range(n_orbs))
    return [single_excitation(full_set - set(string), string, n_elec, n_orbs) for string in strings]

# Generate a functor that transforms ci vector (i.e. operator H in C' = H(C))
def knowles_handy_full_ci_transformer(one_electron_integrals, two_electron_integrals, n_elecs, n_spin=0):
    

    n_rows, n_cols = one_electron_integrals.shape

    n_orbs = n_rows

    assert (n_rows == n_cols)
    assert (np.all(np.array(two_electron_integrals.shape) == n_orbs))

    n_alpha = (n_elecs + n_spin) // 2
    n_beta = (n_elecs - n_spin) // 2

    # This generates all possible combinations of the occupied orbitals, with indices of the orbitals in ascending order
    alpha_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    n_beta_conbinations = len(beta_combinations)

    # the dimension of the hamiltonian matrix (dimension of the determinant basis)
    n_dim = len(alpha_combinations) * len(beta_combinations)

    alpha_single_excitation = single_replacement(alpha_combinations, n_alpha, n_orbs)
    beta_single_excitation = single_replacement(beta_combinations, n_beta, n_orbs)

    # add the original 1e integral and contribution from 2e integral with a delta function \delta_{jk}
    modified_1e_integral = one_electron_integrals - 0.5 * np.einsum("ikkl -> il", two_electron_integrals)

    # define a functor that transforms ci vector linearly
    def transformer(ci_vector):

        # <I | E_ij | J>, for a fixed (ij) it has n_dim elements, or n_string (alpha strings) x n_string (beta strings)
        # Therefore this tensor holds dimension of (n_string x n_string) x n_orbs x n_orbs
        one_particle_matrix = np.zeros((n_dim, n_orbs, n_orbs))

        # This iterates over all possible one-electron excitation from alpha strings
        # alpha_index refers to the index of the original alpha string
        for alpha_index, alpha_excitation_list in enumerate(alpha_single_excitation):
            for alpha_excitation in alpha_excitation_list:
                # This iterates over all the beta strings
                for beta_index in range(len(beta_combinations)):
                    i, j = alpha_excitation["ij"]
                    ci_vector_index = alpha_index + beta_index * n_beta_conbinations
                    one_particle_index = alpha_excitation["det_index"] + beta_index * n_beta_conbinations
                    one_particle_matrix[one_particle_index, i, j] += \
                        alpha_excitation["sign"] * ci_vector[ci_vector_index]

        # This iterates over all possible one-electron excitation from beta strings
        for beta_index, beta_excitation_list in enumerate(beta_single_excitation):
            for beta_excitation in beta_excitation_list:
                for alpha_index in range(len(alpha_combinations)):
                    i, j = beta_excitation["ij"]
                    ci_vector_index = alpha_index + beta_index * n_beta_conbinations
                    one_particle_index = alpha_index + beta_excitation["det_index"] * n_beta_conbinations

                    one_particle_matrix[one_particle_index, i, j] += \
                        beta_excitation["sign"] * ci_vector[ci_vector_index]

        # print("mentor_handy", np.linalg.norm(one_particle_matrix))
        two_electron_contracted = np.einsum("pkl, ijkl -> pij", one_particle_matrix, two_electron_integrals)

        # Start from 1e integral transform
        new_ci_vector = np.einsum("pij, ij -> p", one_particle_matrix, modified_1e_integral)

        for alpha_index, alpha_excitation_list in enumerate(alpha_single_excitation):
            for alpha_excitation in alpha_excitation_list:
                for beta_index in range(len(beta_combinations)):
                    i, j = alpha_excitation["ij"]
                    ci_vector_index = alpha_index + beta_index * n_beta_conbinations
                    one_particle_index = alpha_excitation["det_index"] + beta_index * n_beta_conbinations

                    new_ci_vector[ci_vector_index] += \
                        0.5 * alpha_excitation["sign"] * two_electron_contracted[one_particle_index, i, j]

        for beta_index, beta_excitation_list in enumerate(beta_single_excitation):
            for beta_excitation in beta_excitation_list:
                for alpha_index in range(len(alpha_combinations)):
                    i, j = beta_excitation["ij"]
                    ci_vector_index = alpha_index + beta_index * n_beta_conbinations
                    one_particle_index = alpha_index + beta_excitation["det_index"] * n_beta_conbinations

                    new_ci_vector[ci_vector_index] += \
                        0.5 * beta_excitation["sign"] * two_electron_contracted[one_particle_index, i, j]

        return new_ci_vector

    return transformer

def davidson_diagonalization(transformer,
                             diagonal,
                             eigenvalue_index,
                             start_search_dim,
                             n_dim,
                             residue_tol=1e-5,
                             max_iter=1000):

    search_space = np.eye(n_dim, start_search_dim) + 0.01

    for iter in range(max_iter):
        # print(iter)
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
        # print(np.linalg.norm(residue))
        if np.linalg.norm(residue) < residue_tol:
            return eig, eigvec

        xi = np.dot(np.diagflat(1.0 / (eig - diagonal)), residue)

        np.eye(n_dim) - np.einsum('ij, kj -> jik', orthonormal_subspace, orthonormal_subspace)

        search_space = np.concatenate((orthonormal_subspace, np.array([xi]).T), axis=1)

    raise Exception("Davidson diagonaliztion failed")
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